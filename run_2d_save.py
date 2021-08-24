from __future__ import print_function, absolute_import, division

import datetime
import os
import os.path as path
import random

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn

from function_baseline.config import get_parse_args
from function_baseline.data_preparation_custom import Data_Custom
from function_baseline.model_pos_preparation import model_pos_preparation
from function_baseline.model_pose_train_custom import train
from function_poseaug.model_pos_eval_custom import evaluate
from pelee.lib.models.MOBIS_peleenet import get_pose_pelee_net
from common import get_resnet
from utils.log import Logger, savefig
from utils.utils import save_ckpt
import cv2
"""
this code is used to pretrain the baseline model
1. Simple Baseline
2. VideoPose
3. SemGCN
4. ST-GCN
code are modified from https://github.com/garyzhao/SemGCN
"""
pixel_mean = (0.485, 0.456, 0.406)
pixel_std = (0.229, 0.224, 0.225)

def main(args):
    print('==> Using settings {}'.format(args))
    device = torch.device("cuda")

    print('==> Loading dataset...')
    data_class = Data_Custom(is_train = args.is_train, detection_2d=True)
    loader = data_class.data_preparation(args)
    
    if 'pelee' == args.keypoints:
        print("==> Creating 2D pose estimation model...")
        estimator_2d = get_pose_pelee_net(is_train=False).cuda()
        estimator_2d.load_state_dict(torch.load(args.path_2d, map_location='cpu'))
    elif 'resnet' in args.keypoints:
        print("==> Creating 2D pose estimation model...")
        estimator_2d = get_resnet(args).cuda()
        estimator_2d.load_state_dict(torch.load(args.path_2d, map_location='cpu'))
    estimator_2d.eval()
    
    #################################################
    # ########## start inferencing
    #################################################
    print("==> Inferencing...")
    keypoints_2d = []
    with torch.no_grad():
        # data loading
        for i, temp in enumerate(loader):
            img_patch, joint_img, joint_cam, joint_vis, bbox, img_width, img_height = temp
            img_patch = img_patch.to(device)
            
            # inferencing
            output_heatmaps = estimator_2d(img_patch)
            heatmap_h, heatmap_w = output_heatmaps.shape[2:]
            
            # save_batch_heatmaps(img_patch, output_heatmaps, 'heatmap.jpg') # activate to save heatmap images
            
            # to keypoints
            detected_2d = get_max_preds(output_heatmaps.cpu().numpy())
            detected_2d[:, :, 0] = detected_2d[:, :, 0] * bbox[:, 2:3].cpu().numpy() / heatmap_h + bbox[:, :1].cpu().numpy()
            detected_2d[:, :, 1] = detected_2d[:, :, 1] * bbox[:, 3:].cpu().numpy() / heatmap_w + bbox[:, 1:2].cpu().numpy()
            
            keypoints_2d.append(detected_2d)
            
    keypoints_2d = np.concatenate(keypoints_2d, axis=0)
    
    if args.is_train:
        save_name = args.keypoints + '_train'
    else:
        save_name = args.keypoints + '_valid'
    np.savez_compressed(save_name, x = keypoints_2d[:, :, 0], y = keypoints_2d[:, :, 1])

def get_max_preds(batch_heatmaps):
    '''
    get predictions from score maps
    heatmaps: numpy.ndarray([batch_size, num_joints, height, width])
    '''
    assert isinstance(batch_heatmaps, np.ndarray), \
        'batch_heatmaps should be numpy.ndarray'
    assert batch_heatmaps.ndim == 4, 'batch_images should be 4-ndim'

    batch_size = batch_heatmaps.shape[0]
    num_joints = batch_heatmaps.shape[1]
    width = batch_heatmaps.shape[3]
    heatmaps_reshaped = batch_heatmaps.reshape((batch_size, num_joints, -1))
    idx = np.argmax(heatmaps_reshaped, 2)
    maxvals = np.amax(heatmaps_reshaped, 2)

    maxvals = maxvals.reshape((batch_size, num_joints, 1))
    idx = idx.reshape((batch_size, num_joints, 1))

    preds = np.tile(idx, (1, 1, 2)).astype(np.float32)

    preds[:, :, 0] = (preds[:, :, 0]) % width
    preds[:, :, 1] = np.floor((preds[:, :, 1]) / width)

    pred_mask = np.tile(np.greater(maxvals, 0.0), (1, 1, 2))
    pred_mask = pred_mask.astype(np.float32)

    preds *= pred_mask
    return preds

def normalize_screen_coordinates(X, w, h):
    assert X.shape[-1] == 2

    # Normalize so that [0, w] is mapped to [-1, 1], while preserving the aspect ratio
    return X / w * 2 - [1, h / w]

def save_batch_heatmaps(batch_image, batch_heatmaps, file_name,
                        normalize=True):
    '''
    batch_image: [batch_size, channel, height, width]
    batch_heatmaps: ['batch_size, num_joints, height, width]
    file_name: saved file name
    '''
    if normalize:
        batch_image = batch_image.clone()
        min = float(batch_image.min())
        max = float(batch_image.max())

        batch_image.add_(-min).div_(max - min + 1e-5)

    batch_size = batch_heatmaps.size(0)
    num_joints = batch_heatmaps.size(1)
    heatmap_height = batch_heatmaps.size(2)
    heatmap_width = batch_heatmaps.size(3)

    grid_image = np.zeros((batch_size*heatmap_height,
                        (num_joints+1)*heatmap_width,
                        3),
                        dtype=np.uint8)

    preds = get_max_preds(batch_heatmaps.detach().cpu().numpy())

    for i in range(batch_size):
        image = batch_image[i].mul(255)\
                            .clamp(0, 255)\
                            .byte()\
                            .permute(1, 2, 0)\
                            .cpu().numpy()
        heatmaps = batch_heatmaps[i].mul(255)\
                                    .clamp(0, 255)\
                                    .byte()\
                                    .cpu().numpy()
        image = image[..., ::-1]
        resized_image = cv2.resize(image,
                                (int(heatmap_width), int(heatmap_height)))

        height_begin = heatmap_height * i
        height_end = heatmap_height * (i + 1)
        for j in range(num_joints):
            cv2.circle(resized_image,
                    (int(preds[i][j][0]), int(preds[i][j][1])),
                    1, [0, 0, 255], 1)
            heatmap = heatmaps[j, :, :]
            colored_heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
            masked_image = colored_heatmap*0.7 + resized_image*0.3
            cv2.circle(masked_image,
                    (int(preds[i][j][0]), int(preds[i][j][1])),
                    1, [0, 0, 255], 1)

            width_begin = heatmap_width * (j+1)
            width_end = heatmap_width * (j+2)
            grid_image[height_begin:height_end, width_begin:width_end, :] = \
                masked_image
            # grid_image[height_begin:height_end, width_begin:width_end, :] = \
            #     colored_heatmap*0.7 + resized_image*0.3

        grid_image[height_begin:height_end, 0:heatmap_width, :] = resized_image

    cv2.imwrite(file_name, grid_image)

if __name__ == '__main__':
    args = get_parse_args()
    # fix random
    random_seed = args.random_seed
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    np.random.seed(random_seed)
    random.seed(random_seed)
    os.environ['PYTHONHASHSEED'] = str(random_seed)
    # copy from #https://pytorch.org/docs/stable/notes/randomness.html
    torch.backends.cudnn.deterministic = True
    cudnn.benchmark = True

    main(args)
