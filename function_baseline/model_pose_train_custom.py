
from __future__ import print_function, absolute_import, division

import time

import torch
import torch.nn as nn

from progress.bar import Bar
from utils.utils import AverageMeter, lr_decay
from common.camera import world_to_camera, normalize_screen_coordinates
from data_extra.dataset_converter import COCO2HUMAN, MPII2HUMAN

'''
Code are modified from https://github.com/garyzhao/SemGCN
This train function is adopted from SemGCN for baseline training.
'''

'''
peleenet
0 : nose
1 : left_eye
2 : right_eye
3 : left_ear
4 : right_ear
5 : left_shoulder
6 : right_shoulder
7 : left_elbow
8 : right_elbow
9 : left_wrist
10 : right_wrist
11 : left_hip
12 : right_hip
13 : left_knee
14 : right_knee
15 : left_ankle
16 : right_ankle
'''

def train(data_loader, model_pos, criterion, optimizer, device, lr_init, lr_now, step, decay, gamma, type_2d, max_norm=True, estimator_2d=None):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    epoch_loss_3d_pos = AverageMeter()

    # Switch to train mode
    torch.set_grad_enabled(True)
    model_pos.train()
    end = time.time()

    bar = Bar('Train', max=len(data_loader))
    # for i, (targets_3d, inputs_2d, _, _) in enumerate(data_loader):
    for i, (img_patch, joint_img, joint_cam, joint_vis) in enumerate(data_loader):
        
        # Measure data loading time
        data_time.update(time.time() - end)
        num_poses = joint_cam.size(0)


        step += 1
        if step % decay == 0 or step == 1:
            lr_now = lr_decay(optimizer, step, lr_init, decay, gamma)

        img_patch, joint_img, joint_cam = img_patch.to(device), joint_img.to(device), joint_cam.to(device)
        targets_3d = joint_cam[:, :, :] - joint_cam[:, :1, :]  # the output is relative to the 0 joint
        
        if type_2d == 'gt':
            inputs_2d = joint_img[:, :, :2]
        else:
            assert estimator_2d is not None
            with torch.no_grad():
                heatmaps_2d = estimator_2d(img_patch)
                B, C, H, W = heatmaps_2d.shape
                heatmaps_2d = heatmaps_2d.view((B, C, -1)).contiguous()
                heatmap_max = heatmaps_2d.max(dim=-1)[1]
                x_points = heatmap_max % W; y_points = heatmap_max // W
                width_ratio = img_patch.shape[3] / W; height_ratio = img_patch.shape[2]
                keypoints_2d = torch.cat([x_points.unsqueeze(2) * width_ratio , y_points.unsqueeze(2) * height_ratio], dim=2)
                if type_2d == 'pelee':
                    inputs_2d = COCO2HUMAN(keypoints_2d).cuda()
                elif type_2d == 'resnet':
                    inputs_2d = MPII2HUMAN(keypoints_2d).cuda()
                else:
                    raise NotImplementedError("Corresponding network is not supported")
                        
                
        outputs_3d = model_pos(inputs_2d)

        optimizer.zero_grad()
        loss_3d_pos = criterion(outputs_3d, targets_3d)
        loss_3d_pos.backward()
        if max_norm:
            nn.utils.clip_grad_norm_(model_pos.parameters(), max_norm=1)
        optimizer.step()

        epoch_loss_3d_pos.update(loss_3d_pos.item(), num_poses)

        # Measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        bar.suffix = '({batch}/{size}) Data: {data:.6f}s | Batch: {bt:.3f}s | Total: {ttl:} | ETA: {eta:} ' \
                    '| Loss: {loss: .4f}' \
            .format(batch=i + 1, size=len(data_loader), data=data_time.avg, bt=batch_time.avg,
                    ttl=bar.elapsed_td, eta=bar.eta_td, loss=epoch_loss_3d_pos.avg)
        bar.next()

    bar.finish()
    return epoch_loss_3d_pos.avg, lr_now, step


