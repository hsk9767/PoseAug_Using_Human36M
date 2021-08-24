from __future__ import print_function, absolute_import, division

import datetime
import os
import os.path as path
import random
import time
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
from torch.nn.functional import bilinear

from function_baseline.config import get_parse_args
from function_baseline.data_preparation_custom import Data_Custom
from function_baseline.model_pos_preparation import model_pos_preparation
from one_stage import get_pose_net
from pelee.lib.models.MOBIS_peleenet import get_pose_pelee_net
from common import get_resnet
from common.viz import show_3d_moon
from common.common_dataset import DatasetLoader_3d_mppe
from data_extra.dataset_converter import COCO2HUMAN, MPII2HUMAN
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
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
    assert args.batch_size == 1
    # include
    path_3d = 'common.' + 'h36m_dataset_custom'
    exec('from ' + path_3d + ' import ' + 'Human36M')
    
    # loader
    if args.keypoints == 'one_stage':
        dataset_3d = DatasetLoader_3d_mppe(eval('Human36M')('vis', True), ref_joints_name=None, is_train=False, transform=transforms.Compose([\
                                                                                                                        transforms.ToTensor()
                                                                                                                        , transforms.Normalize(mean=pixel_mean, std=pixel_std)]), vis=True)
        loader = DataLoader(dataset_3d, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    elif ('pelee' == args.keypoints) or ('resnet' in args.keypoints):
        data_class = Data_Custom(vis=True, detection_2d=True, original=False)
        loader = data_class.data_preparation(args)
    
    # activate lifting 
    if args.keypoints != 'one_stage':
        print("==> Creating model...")
        model_pos = model_pos_preparation(args, device).cuda()
        assert path.isfile(args.evaluate), '==> No checkpoint found at {}'.format(args.evaluate)
        print("==> Loading checkpoint '{}'".format(args.evaluate))
        ckpt = torch.load(args.evaluate)
        model_pos.eval()
        model_pos.load_state_dict(ckpt['state_dict'])
    
    
    # activate one-stage model or lifting model
    if 'pelee' == args.keypoints:
        print("==> Creating 2D pose estimation model...")
        estimator_2d = get_pose_pelee_net(is_train=False).cuda()
        estimator_2d.load_state_dict(torch.load(args.path_2d, map_location='cpu'))
        estimator_2d.eval()
    elif 'resnet' in args.keypoints:
        print("==> Creating 2D pose estimation model...")
        estimator_2d = get_resnet(args).cuda()
        estimator_2d.load_state_dict(torch.load(args.path_2d, map_location='cpu'))
        estimator_2d.eval()
    elif 'one_stage' == args.keypoints:
        one_stage_model = get_pose_net(50, is_train=False, joint_num=18).cuda()
        one_stage_model = torch.nn.DataParallel(one_stage_model)
        one_stage_model.load_state_dict(torch.load(args.path_one_stage)['network'])
        one_stage_model.eval()
    else:
        raise NotImplementedError("Not supported")
    
    save_path = path.join(args.vis_save_path)
    os.makedirs(save_path, exist_ok=True)
    print('==> Making checkpoint dir: {}'.format(save_path))

    #################################################
    ################start inferencing################
    #################################################
    print("==> Inferencing...")
    
    with torch.no_grad():
        # data loading
        save_paths = []
        for i, temp in enumerate(loader):
            if args.keypoints == 'one_stage':
                raw_img_path, img_patch, bbox, f, c, root_cam = temp
                img_patch, bbox, f, c, root_cam = img_patch.to(device), bbox.to(device), f.to(device), c.to(device), root_cam.to(device)
            else:
                raw_img_path, img_patch, bbox = temp
                img_patch, bbox = img_patch.to(device), bbox.to(device)
            if not(args.what_to_vis in raw_img_path[0]):
                continue
            # print('raw_img_path : ', raw_img_path[0])
            
            raw_img = cv2.imread(raw_img_path[0], cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
            for ch in range(3):
                raw_img[:, :, ch] = np.clip(raw_img[:, :, ch], 0, 255)
            img_h, img_w = raw_img.shape[:2]
            # to check the joint_img input
            # outputs_3d = model_pos(joint_img)
            # outputs_3d = outputs_3d[:, :, :] - outputs_3d[:, :1, :]

            # 1-stage
            if 'one_stage' == args.keypoints:
                start_time = time.time()
                output_coord = one_stage_model(img_patch)
                torch.cuda.synchronize()
                taken_time = time.time() - start_time
                output_coord[:, :, 0] = output_coord[:, :, 0] / 64 * bbox[0][2] + bbox[0][0]
                output_coord[:, :, 1] = output_coord[:, :, 1] / 64 * bbox[0][3] + bbox[0][1]
                output_coord[:, :, 2] = (output_coord[:, :, 2] / 64 * 2 - 1) * (1000) + root_cam[0][2]
                output_coord[:, :, 0] = (output_coord[:, :, 0] - c[0][0]) / f[0][0] * output_coord[0, :, 2]
                output_coord[:, :, 1] = (output_coord[:, :, 1] - c[0][1]) / f[0][1] * output_coord[0, :, 2]
                outputs_3d = output_coord[:, :, :] - output_coord[:, :1, :]
                outputs_3d = outputs_3d / 1000.
                
                fps = str(1 / taken_time)[:6]
                print('fps : ', fps)

            # 2-stage
            if ('pelee' == args.keypoints) or ('resnet' in args.keypoints):
                start_time = time.time()
                output_heatmaps = estimator_2d(img_patch)
                heatmap_h, heatmap_w = output_heatmaps.shape[-2:]
                outputs_2d = get_max_preds(output_heatmaps.cpu().numpy())
                outputs_2d[:, :, 0] = outputs_2d[:, :, 0] * bbox[:, 2:3].cpu().numpy() / heatmap_h + bbox[:, :1].cpu().numpy()
                outputs_2d[:, :, 1] = outputs_2d[:, :, 1] * bbox[:, 3:].cpu().numpy() / heatmap_w + bbox[:, 1:2].cpu().numpy()
                if 'pelee' == args.keypoints:
                    outputs_2d = COCO2HUMAN(outputs_2d.copy())
                else:
                    outputs_2d = MPII2HUMAN(outputs_2d.copy())
                outputs_2d = normalize_screen_coordinates(outputs_2d, img_w, img_h).astype(np.float32)
                outputs_2d = torch.from_numpy(outputs_2d).cuda()
                
                outputs_3d = model_pos(outputs_2d)
                torch.cuda.synchronize()
                taken_time = time.time() - start_time
                fps = str(1 / taken_time)[:6]
                print('fps : ', fps)
                outputs_3d = outputs_3d[:, :, :] - outputs_3d[:, :1, :]
            
            # visualize
            path_ = raw_img_path[0].split('_')[-1]
            save_path = path.join(args.vis_save_path, path_)
            show_3d_moon(outputs_3d.cpu().numpy()[0], loader.dataset.skeleton, save_path)
            
            # with raw image
            raw_img_resized = cv2.resize(raw_img, (500, 500), interpolation=1)
            
            skeleton_img = cv2.imread(save_path)
            skeleton_img_resized = cv2.resize(skeleton_img,  (500, 500), interpolation=1)
            
            save_img = np.concatenate([raw_img_resized, skeleton_img_resized], axis=1)
            save_path = path.join(args.vis_save_path, path_)
            save_paths.append(save_path)
            cv2.putText(save_img, 'FPS : ' + fps, (30, 30), cv2.FONT_HERSHEY_COMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)
            cv2.imwrite(save_path, save_img)
            
        
        # to_video
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_save_path = path.join(args.vis_save_path, args.keypoints + '_' + 'video.mp4')
        video_out = cv2.VideoWriter(video_save_path, fourcc, 30, (1000, 500))
        for p in save_paths:
            to_save = cv2.imread(p)
            video_out.write(to_save)
        video_out.release()

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
