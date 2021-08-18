import os, sys 
sys.path.append('..')
import torch
from hrnet_simplebaseline.lib.config import cfg, update_config
from hrnet_simplebaseline.lib.models.pose_hrnet import get_pose_net as get_hrnet_model
from hrnet_simplebaseline.lib.models.pose_resnet import get_pose_net as get_resnet_model

def get_resnet(args):
    mode = args.keypoints.split('_')[-1]
    assert mode in ['50', '101', '152']
    
    if mode == '50':
        cfg_file = 'hrnet_simplebaseline/experiments/mpii/resnet/res50_256x256_d256x3_adam_lr1e-3.yaml'
    elif mode == '101':
        cfg_file = 'hrnet_simplebaseline/experiments/mpii/resnet/res101_256x256_d256x3_adam_lr1e-3.yaml'
    elif mode == '152':
        cfg_file = 'hrnet_simplebaseline/experiments/mpii/resnet/res152_256x256_d256x3_adam_lr1e-3.yaml'
    dict_ = {'cfg' : cfg_file, 
        'opts' : None}
    update_config(cfg, dict_)
    model = get_resnet_model(cfg, is_train=False)
    
    return model
    


