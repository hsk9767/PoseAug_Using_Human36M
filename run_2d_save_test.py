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
from torch.utils.data import DataLoader
from data_extra.dataset_converter import COCO2HUMAN, MPII2HUMAN
from common.common_dataset import DatasetLoader_saved_test
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import cv2
pixel_mean = (0.485, 0.456, 0.406)
pixel_std = (0.229, 0.224, 0.225)

"""
this code is used to pretrain the baseline model
1. Simple Baseline
2. VideoPose
3. SemGCN
4. ST-GCN
code are modified from https://github.com/garyzhao/SemGCN
"""


def main(args):
    keypoints_saved = np.load(f'{args.keypoints}_valid.npz')
    x_2d = np.expand_dims(keypoints_saved['x'], 2)
    y_2d = np.expand_dims(keypoints_saved['y'], 2)
    keypoints_saved = np.concatenate([x_2d, y_2d], 2)
    if 'resnet' in args.keypoints:
        keypoints_saved = MPII2HUMAN(keypoints_saved.copy())
    elif 'pelee' in args.keypoints:
        keypoints_saved = COCO2HUMAN(keypoints_saved.copy())
    
    print('==> Using settings {}'.format(args))
    device = torch.device("cuda")

    print('==> Loading dataset...')
    path_3d = 'common.' + 'h36m_dataset_custom'
    exec('from ' + path_3d + ' import ' + 'Human36M')
    dataset_3d = DatasetLoader_saved_test(eval('Human36M')('test'), ref_joints_name=None, is_train=True, transform=transforms.Compose([\
                                                                                                                            transforms.ToTensor(),
                                                                                                                            transforms.Normalize(mean=pixel_mean, std=pixel_std)]), detection_2d=True)
    loader = DataLoader(dataset_3d, batch_size=1, shuffle=False, num_workers=2, pin_memory=True)
    
    print('==> Visualization preparation...')
    cmap = plt.get_cmap('rainbow')
    colors = [cmap(i) for i in np.linspace(0, 1, 16+2)]
    colors = [(c[2] * 255, c[1] * 255, c[0] * 255) for c in colors ]
    
    #################################################
    # ########## start inferencing
    #################################################
    print("==> Inferencing...")
    with torch.no_grad():
        # data loading
        for i, temp in enumerate(loader):
            if i == 10:
                break
            
            img, joint_img, bbox, img_width, img_height = temp
            
            # GT 2D
            img_ = img[0].clone().numpy()
            temporal_keypoint = joint_img[0].numpy() #first in the mini_batch

            for j in range(16):
                joint_keypoint = (temporal_keypoint[j][0], temporal_keypoint[j][1])
                cv2.circle(
                    img_, joint_keypoint, color = colors[j], thickness=-1, radius=2
                )
            cv2.imwrite(f'save_test/{i}_GT.jpg', img_)
            
            # saved 2D
            img__ = img[0].clone().numpy()
            temporal_keypoint = keypoints_saved[i].astype(np.int32)
            
            for j in range(16):
                joint_keypoint = (temporal_keypoint[j][0], temporal_keypoint[j][1])
                cv2.circle(
                    img__, joint_keypoint, color = colors[j], thickness=-1, radius=2
                )
            cv2.imwrite(f'save_test/{i}_{args.keypoints}.jpg', img__)
    print("==> Done...!")

def get_max_preds(batch_heatmaps, bbox):
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
    
    # to coordinates in the whole image
    bbox = np.expand_dims(bbox[:, :2], 1)
    preds += bbox
    
    return preds

def normalize_screen_coordinates(X, w, h):
    assert X.shape[-1] == 2

    # Normalize so that [0, w] is mapped to [-1, 1], while preserving the aspect ratio
    return X / w * 2 - [1, h / w]

def de_normalize_screen_coordinates(X, w, h):
    assert X.shape[-1] == 2
    return (X + [1, h/w]) * w / 2

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
