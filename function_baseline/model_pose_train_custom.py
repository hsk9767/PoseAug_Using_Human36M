
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

def train(data_loader, model_pos, criterion, optimizer, device, lr_init, lr_now, step, decay, gamma, max_norm=True):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    epoch_loss_3d_pos = AverageMeter()

    # Switch to train mode
    torch.set_grad_enabled(True)
    model_pos.train()
    end = time.time()

    bar = Bar('Train', max=len(data_loader))
    # for i, (targets_3d, inputs_2d, _, _) in enumerate(data_loader):
    for i, (joint_img, joint_cam, joint_vis) in enumerate(data_loader):
        
        # Measure data loading time
        data_time.update(time.time() - end)
        num_poses = joint_cam.size(0)

        step += 1
        if (step % decay == 0 or step == 1) and (lr_now >= 1e-6):
            lr_now = lr_decay(optimizer, step, lr_init, decay, gamma)

        joint_img, joint_cam = joint_img.to(device), joint_cam.to(device)
        targets_3d = joint_cam[:, :, :] - joint_cam[:, :1, :]  # the output is relative to the 0th joint
        
        inputs_2d = joint_img[:, :, :2]
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


def train_3d_mppe(data_loader, model_pos, criterion, optimizer, device, lr_init, lr_now, step, decay, gamma, max_norm=True):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    epoch_loss_3d_pos = AverageMeter()

    # Switch to train mode
    torch.set_grad_enabled(True)
    model_pos.train()
    end = time.time()

    bar = Bar('Train', max=len(data_loader))
    # for i, (targets_3d, inputs_2d, _, _) in enumerate(data_loader):
    for i, (joint_img, joint_cam, joint_vis, joint_have_depth) in enumerate(data_loader):
        
        # Measure data loading time
        data_time.update(time.time() - end)
        num_poses = joint_cam.size(0)

        step += 1
        if (step % decay == 0 or step == 1) and (lr_now >= 1e-6):
            lr_now = lr_decay(optimizer, step, lr_init, decay, gamma)

        joint_img, joint_cam = joint_img.to(device), joint_cam.to(device)
        targets_3d = joint_cam[:, :, :] - joint_cam[:, :1, :]  # the output is relative to the 0th joint
        
        inputs_2d = joint_img[:, :, :2]
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


