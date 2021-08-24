from __future__ import print_function, absolute_import, division

import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from common.data_loader import PoseDataSet
from progress.bar import Bar
from utils.data_utils import fetch
from utils.loss import mpjpe, p_mpjpe, compute_PCK, compute_AUC
from utils.utils import AverageMeter
from data_extra.dataset_converter import COCO2HUMAN, MPII2HUMAN
from tqdm import tqdm

####################################################################
# ### evaluate p1 p2 pck auc dataset with test-flip-augmentation
####################################################################
def evaluate(data_loader, model_pos_eval, device, keypoints='gt', summary=None, writer=None, key='', tag='', flipaug=''):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    epoch_p1 = AverageMeter()
    epoch_p2 = AverageMeter()

    # Switch to evaluate mode
    model_pos_eval.eval()
    end = time.time()

    bar = Bar('Eval posenet on {}'.format(key), max=len(data_loader))
    for i, temp in enumerate(data_loader):
        joint_img, targets_3d = temp[0], temp[1]
        
        # Measure data loading time
        data_time.update(time.time() - end)
        num_poses = targets_3d.size(0)
        inputs_2d = joint_img[:, :, :2].to(device)

        with torch.no_grad():
            if flipaug:  # flip the 2D pose Left <-> Right
                joints_left = [4, 5, 6, 9, 10, 11]
                joints_right = [1, 2, 3, 12, 13, 14]
                out_left = [4, 5, 6, 9, 10, 11]
                out_right = [1, 2, 3, 12, 13, 14]

                inputs_2d_flip = inputs_2d.detach().clone()
                inputs_2d_flip[:, :, 0] *= -1
                inputs_2d_flip[:, joints_left + joints_right, :] = inputs_2d_flip[:, joints_right + joints_left, :]
                outputs_3d_flip = model_pos_eval(inputs_2d_flip.view(num_poses, -1)).view(num_poses, -1, 3).cpu()
                outputs_3d_flip[:, :, 0] *= -1
                outputs_3d_flip[:, out_left + out_right, :] = outputs_3d_flip[:, out_right + out_left, :]

                outputs_3d = model_pos_eval(inputs_2d.view(num_poses, -1)).view(num_poses, -1, 3).cpu()
                outputs_3d = (outputs_3d + outputs_3d_flip) / 2.0

            else:
                outputs_3d = model_pos_eval(inputs_2d.view(num_poses, -1)).view(num_poses, -1, 3).cpu()

        # caculate the relative position.
        targets_3d = targets_3d[:, :, :] - targets_3d[:, :1, :]  # the output is relative to the 0 joint
        outputs_3d = outputs_3d[:, :, :] - outputs_3d[:, :1, :]
        
        
        p1score = mpjpe(outputs_3d, targets_3d).item() * 1000.0
        epoch_p1.update(p1score, num_poses)
        p2score = p_mpjpe(outputs_3d.numpy(), targets_3d.numpy()).item() * 1000.0
        epoch_p2.update(p2score, num_poses)

        # Measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        bar.suffix = '({batch}/{size}) Data: {data:.6f}s | Batch: {bt:.3f}s | Total: {ttl:} | ETA: {eta:} ' \
                     '| MPJPE: {e1: .4f} | P-MPJPE: {e2: .4f}' \
            .format(batch=i + 1, size=len(data_loader), data=data_time.avg, bt=batch_time.avg,
                    ttl=bar.elapsed_td, eta=bar.eta_td, e1=epoch_p1.avg,
                    e2=epoch_p2.avg)
        bar.next()

    if writer:
        writer.add_scalar('posenet_{}'.format(key) + flipaug + '/p1score' + tag, epoch_p1.avg, summary.epoch)
        writer.add_scalar('posenet_{}'.format(key) + flipaug + '/p2score' + tag, epoch_p2.avg, summary.epoch)

    bar.finish()
    return epoch_p1.avg, epoch_p2.avg


def evaluate_3d_mppe(data_loader, model_pos_eval, device, summary=None, writer=None, key='', tag='', flipaug=''):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    epoch_p1 = AverageMeter()
    epoch_p2 = AverageMeter()

    # Switch to evaluate mode
    model_pos_eval.eval()
    end = time.time()
    
    output = []
    target = []
    with torch.no_grad():
        for i, temp in tqdm(enumerate(data_loader)):
            img_patch, targets_3d, bbox, f, c, root_cam = temp
            bbox, root_cam = bbox.to(device), root_cam.to(device)
            # inferencing
            output_coord = model_pos_eval(img_patch)
            num_batch = output_coord.shape[0]
            # to original coordinate
            for j in range(num_batch):
                output_coord[j, :, 0] = output_coord[j, :, 0] / 64 * bbox[j][2] + bbox[j][0]
                output_coord[j, :, 1] = output_coord[j, :, 1] / 64 * bbox[j][3] + bbox[j][1]
                output_coord[j, :, 2] = (output_coord[j, :, 2] / 64 * 2 -1) * (1000) + root_cam[j][2]
            #to camera coordinates
            for j in range(num_batch):
                output_coord[j, :, 0] = (output_coord[j, :, 0] - c[j][0]) / f[j][0] * output_coord[j, :, 2]
                output_coord[j, :, 1] = (output_coord[j, :, 1] - c[j][1]) / f[j][1] * output_coord[j, :, 2]
            
            # caculate the relative position.
            targets_3d = targets_3d[:, :, :] - targets_3d[:, :1, :] # the output is relative to the root joint
            outputs_3d = output_coord[:, :, :] - output_coord[:, :1, :]
            
            # w/o thorax
            targets_3d = targets_3d[:, :-1, :] 
            outputs_3d = outputs_3d[:, :-1, :]
            
            output.append(outputs_3d.cpu().numpy())
            target.append(targets_3d.cpu().numpy())
    print('==> Pre-processing end')
    output = np.concatenate(output, axis=0)
    target = np.concatenate(target, axis=0)
    
    # error calculate
    error = []
    for i in range(len(output)):
        error.append(np.sqrt(np.sum((output[i] - target[i])**2,1)))
    
    print(f'MPJPE(mm) : {np.mean(error)}')
    return np.mean(error)
#########################################
# overall evaluation function
#########################################
def evaluate_posenet(args, data_dict, model_pos, model_pos_eval, device, summary, writer, tag):
    """
    evaluate H36M and 3DHP
    test-augment-flip only used for 3DHP as it does not help on H36M.
    """
    with torch.no_grad():
        model_pos_eval.load_state_dict(model_pos.state_dict())
        h36m_p1, h36m_p2 = evaluate(data_dict['valid_loader'], model_pos_eval, device, summary, writer,
                                             key='H36M_test', tag=tag, flipaug='')  # no flip aug for h36m
        # dhp_p1, dhp_p2 = evaluate(data_dict['mpi3d_loader'], model_pos_eval, device, summary, writer,
        #                                    key='mpi3d_loader', tag=tag, flipaug='_flip')
    return h36m_p1, h36m_p2#, dhp_p1, dhp_p2

