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
from function_finetune import soft_argmax
from pelee.lib.models.MOBIS_peleenet import get_pose_pelee_net
from common import get_resnet
from progress.bar import Bar


"""
this code is used to pretrain the baseline model
1. Simple Baseline
2. VideoPose
3. SemGCN
4. ST-GCN
code are modified from https://github.com/garyzhao/SemGCN
"""


def main(args):
    print('==> Using settings {}'.format(args))
    device = torch.device("cuda")

    print('==> Loading dataset...')
    data_class = Data_Custom(is_train=True, with_valid=True, detection_2d=True)
    data_dict = data_class.data_preparation(args)
    
    print("==> Creating 2D pose estimation model...")    
    if 'pelee' == args.keypoints:
        estimator_2d = get_pose_pelee_net(is_train=True).cuda()
    elif 'resnet' in args.keypoints:
        print("==> Creating 2D pose estimation model...")
        estimator_2d = get_resnet(args).cuda()
    else:
        raise NotImplementedError("Not supported 2D networks")
    estimator_2d = nn.DataParallel(estimator_2d)
    print("==> Prepare optimizer...")
    criterion = nn.MSELoss(reduction='mean').to(device)
    optimizer = torch.optim.Adam(estimator_2d.parameters(), lr=0.001)
    
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, [90, 110], 0.1,
        last_epoch=-1
    )

    ckpt_dir_path = path.join('data', 'Human3.6M', 'Fine_Tune', args.keypoints)
    os.makedirs(ckpt_dir_path, exist_ok=True)
    print('==> Making checkpoint dir: {}'.format(ckpt_dir_path))

    best_perf = 10000000.
    for epoch in range(140):
        # train
        print(f'\n Epoch : {epoch}')
        bar = Bar('Train', max=len(data_dict['train_loader']))
        estimator_2d.train()
        for i, temp in enumerate(data_dict['train_loader']):
            train_image, joint_img = temp[0], temp[1]
            # loading
            train_image, joint_img = train_image.to(device), joint_img.to(device)
            # inference
            output_heatmaps = estimator_2d(train_image)
            predicted_2d = soft_argmax(output_heatmaps)
            # loss
            coord_loss = criterion(joint_img, predicted_2d)
            # backward
            optimizer.zero_grad()
            coord_loss.backward()
            optimizer.step()
            #printing
            bar.suffix = '({batch} / {size})'.format(batch=i+1, size=len(data_dict['train_loader'])) 
            bar.next()
        
        # evaluate
        with torch.no_grad():
            estimator_2d.eval()
            val_loss = 0.
            
            for j, temp in enumerate(data_dict['valid_loader']):
                valid_image, joint_img = temp[0].to(device), temp[1].to(device)
                output_heatmaps = estimator_2d(valid_image)
                predicted_2d = soft_argmax(output_heatmaps)
                coord_loss = criterion(joint_img, predicted_2d)
                val_loss += coord_loss
            val_loss /= j
            if val_loss < best_perf:
                best_perf = val_loss
                state = {
                    'epoch' : epoch,
                    'state_dict' : estimator_2d.state_dict(),
                    'optimizer' : optimizer.state_dict(),
                    'scheduler' : lr_scheduler.state_dict(),
                }
                torch.save(state, path.join(ckpt_dir_path, 'best.pth.tar'))
                print(f'Best model updated in epoch {epoch}')
        # lr_scheduler
        lr_scheduler.step()
    
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
