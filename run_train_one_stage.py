from __future__ import print_function, absolute_import, division

import os
import os.path as path
import random
import math
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from function_baseline.config import get_parse_args
from function_poseaug.model_pos_eval_custom import evaluate, evaluate_3d_mppe
from one_stage import get_pose_net
# from function_poseaug.model_pos_eval import evaluate
from common.common_dataset import DatasetLoader_3d_mppe, MultipleDatasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from utils.utils import AverageMeter
from progress.bar import Bar
import time

pixel_mean = (0.485, 0.456, 0.406)
pixel_std = (0.229, 0.224, 0.225)

def main(args):    
    print('==> Using settings {}'.format(args))
    cudnn.benchmark = True
    device = torch.device("cuda")

    print('==> Loading dataset...')
    # 3d
    path_3d = 'common.' + 'h36m_dataset_custom'
    exec('from ' + path_3d + ' import ' + 'Human36M')
    train_dataset_3d = DatasetLoader_3d_mppe(eval('Human36M')('train', True), ref_joints_name=None, is_train=True, transform=transforms.Compose([\
                                                                                                                        transforms.ToTensor()
                                                                                                                        , transforms.Normalize(mean=pixel_mean, std=pixel_std)]))
    
    
    valid_dataset_3d = DatasetLoader_3d_mppe(eval('Human36M')('test', True), ref_joints_name=None, is_train=False, transform=transforms.Compose([\
                                                                                                                        transforms.ToTensor()
                                                                                                                        , transforms.Normalize(mean=pixel_mean, std=pixel_std)]))
    ref_joints_name = train_dataset_3d.joints_name

    # 2d
    path_3d = 'common.' + 'MPII'
    exec('from ' + path_3d + ' import ' + 'MPII')
    train_dataset_2d = DatasetLoader_3d_mppe(eval("MPII")("train"), ref_joints_name, True, transforms.Compose([\
                                                                                                        transforms.ToTensor(),
                                                                                                        transforms.Normalize(mean=pixel_mean, std=pixel_std)]\
                                                                                                        ))
    
    trainset_3d_loader = MultipleDatasets([train_dataset_3d], make_same_len=False)
    trainset_2d_loader = MultipleDatasets([train_dataset_2d], make_same_len=False)
    trainset_loader = MultipleDatasets([trainset_3d_loader, trainset_2d_loader], make_same_len=True)
    
    train_loader = DataLoader(dataset=trainset_loader, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    valid_loader = DataLoader(valid_dataset_3d, batch_size=32, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    # train_ = DataLoader(train_dataset_3d, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    
    print("==> Creating model...")
    model = get_pose_net(50, is_train=True, joint_num=18).cuda()
    model = torch.nn.DataParallel(model)
    if args.one_stage_continue_train:
        state_dict = torch.load(args.path_one_stage, map_location='cpu')
        model.load_state_dict(state_dict['network'])
        cur_lr = state_dict['lr']
        print(f"==> Loading from {args.path_one_stage}")
    
    print("==> Prepare optimizer...")    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    if args.one_stage_continue_train:
        for g in optimizer.param_groups:
            g['lr'] = cur_lr
    
    ckpt_dir_path = path.join('data', 'Human3.6M', 'one_stage')
    os.makedirs(ckpt_dir_path, exist_ok=True)
    print('==> Making checkpoint dir: {}'.format(ckpt_dir_path))
    
    best_perf = 10000000.
    for epoch in range(25):
        batch_time = AverageMeter()
        train_loss = AverageMeter()
        end_time = time.time()
        
        # lr setting
        decay_epoch = [args.dec_start, args.dec_end]
        for e in decay_epoch:
            if epoch < e:
                break
        if epoch < args.dec_end:
            idx = decay_epoch.index(e)
            for g in optimizer.param_groups:
                g['lr'] = args.lr / (args.dec_fac ** idx)
        else:
            for g in optimizer.param_groups:
                g['lr'] = args.lr / (args.dec_fac ** 2)
        
        print(f'\n Epoch : {epoch}')
        bar = Bar('Train', max=len(train_loader))
        
        # train
        model.train()
        for i, (img_patch, joint_img, joint_vis, joints_have_depth) in enumerate(train_loader):
            # data loading
            img_patch, joint_img, joint_vis, joints_have_depth = \
                img_patch.to(device), joint_img.to(device), joint_vis.to(device), joints_have_depth.to(device)
            
            # forwarding
            target = {'coord': joint_img, 'vis': joint_vis, 'have_depth': joints_have_depth}
            loss_coord = model(img_patch, target)
            
            # back-propagation
            optimizer.zero_grad()
            loss_coord.mean().backward()
            optimizer.step()
            
            #printing
            train_loss.update(loss_coord.mean())
            batch_time.update(time.time() - end_time)
            end_time = time.time()
            bar.suffix = '({batch} / {size})    |    Loss(avg) : {loss}    |    batch_time : {time}'.format(batch=i+1, size=len(train_loader), loss=train_loss.avg, time = batch_time.avg)
            bar.next()
            
        
        # evaluate
        with torch.no_grad():
            error = evaluate_3d_mppe(valid_loader, model, device)
        
        # save
        if best_perf > error:
            best_perf = error
            
            for g in optimizer.param_groups:
                cur_lr = g['lr']
            state = {
                'epoch' : epoch,
                'state_dict' : model.state_dict(), 
                'optimizer' : optimizer.state_dict(),
                "lr" : cur_lr
            }
            
            torch.save(state, path.join(ckpt_dir_path, f'{epoch}_pth.tar'))
            print('==> Best model Updated!')

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
    cudnn.fastest = True
    cudnn.benchmark = True

    main(args)