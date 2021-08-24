from __future__ import print_function, absolute_import, division
import sys
import os.path as path

import numpy as np
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

from common.common_dataset import DatasetLoader, DatasetLoader_only_lifting

pixel_mean = (0.485, 0.456, 0.406)
pixel_std = (0.229, 0.224, 0.225)

class Data_Custom(object):
    def __init__(self, is_train=True, with_valid = False, detection_2d = False, vis=False, original=False):
        super().__init__()
        self.vis = vis
        self.finetune = with_valid
        self.is_train = is_train
        self.detection_2d_save = detection_2d
        self.original = original
        
    def data_preparation(self, args):
        print('==> Data preparation using 3DMPPE...')
        
        path_3d = 'common.' + 'h36m_dataset_custom'
        exec('from ' + path_3d + ' import ' + 'Human36M')
        
        # for just visualizing
        if self.vis:
            if self.original:
                dataset_3d = DatasetLoader(eval('Human36M')('vis', True), ref_joints_name=None, is_train=False, transform=transforms.Compose([\
                                                                                                                        transforms.ToTensor()
                                                                                                                    , transforms.Normalize(mean=pixel_mean, std=pixel_std)]), vis=self.vis, detection_2d = self.detection_2d_save, args=args)
            else:
                dataset_3d = DatasetLoader(eval('Human36M')('vis'), ref_joints_name=None, is_train=False, transform=transforms.Compose([\
                                                                                                                            transforms.ToTensor()
                                                                                                                            , transforms.Normalize(mean=pixel_mean, std=pixel_std)]), vis=self.vis, detection_2d = self.detection_2d_save, args=args)
            
            loader = DataLoader(dataset_3d, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
            return loader
        # to save the results of the 2D detector
        if self.detection_2d_save:
            if self.is_train:
                dataset_3d = DatasetLoader(eval('Human36M')('train'), ref_joints_name=None, is_train=True, transform=transforms.Compose([\
                                                                                                                        transforms.ToTensor(),
                                                                                                                        transforms.Normalize(mean=pixel_mean, std=pixel_std)]), detection_2d=True, finetune=self.finetune)
            else:
                dataset_3d = DatasetLoader(eval('Human36M')('test'), ref_joints_name=None, is_train=False, transform=transforms.Compose([\
                                                                                                                        transforms.ToTensor()
                                                                                                                        , transforms.Normalize(mean=pixel_mean, std=pixel_std)]), detection_2d=True)
                
            # finetune -> dataloader.shuffle = True
            if self.finetune:
                train_loader = DataLoader(dataset_3d, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
                valid_dataset_3d = DatasetLoader(eval('Human36M')('test'), ref_joints_name=None, is_train=False, transform=transforms.Compose([\
                                                                                                                        transforms.ToTensor()
                                                                                                                        , transforms.Normalize(mean=pixel_mean, std=pixel_std)]), detection_2d=True, finetune=self.finetune)
                valid_loader = DataLoader(valid_dataset_3d, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
                return {
                    'train_loader' : train_loader,
                    'valid_loader' : valid_loader
                }
            else:
                train_loader = DataLoader(dataset_3d, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
            return train_loader

        # train -> train loader & valid loader , valid -> None & valid_laoder
        if self.is_train:
            train_dataset_3d = DatasetLoader_only_lifting(eval('Human36M')('train'), ref_joints_name=None, is_train=True, transform=transforms.Compose([\
                                                                                                                    transforms.ToTensor(),
                                                                                                                    transforms.Normalize(mean=pixel_mean, std=pixel_std)]), keypoints=args.keypoints)
        else:
            train_dataset_3d = None

        valid_dataset_3d = DatasetLoader_only_lifting(eval('Human36M')('test'), ref_joints_name=None, is_train=False, transform=transforms.Compose([\
                                                                                                            transforms.ToTensor(),
                                                                                                                    transforms.Normalize(mean=pixel_mean, std=pixel_std)]), keypoints=args.keypoints)
        if self.is_train:
            train_loader = DataLoader(train_dataset_3d, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
        else:
            train_loader = None
        valid_loader = DataLoader(valid_dataset_3d, batch_size=int(args.batch_size / 8), shuffle=False, num_workers=args.num_workers, pin_memory=True)
        
        return {
            'train_loader' : train_loader,
            'valid_loader' : valid_loader
        }