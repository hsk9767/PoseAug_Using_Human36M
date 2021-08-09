import os, sys 
sys.path.append('..')
import torch
from hrnet_simplebaseline.lib.config import cfg, update_config
from hrnet_simplebaseline.lib.models.pose_hrnet import get_pose_net as get_hernet_model
from hrnet_simplebaseline.lib.models.pose_resnet import get_pose_net as get_resnet_model

def get_resnet(args):
    dict_ = {'cfg' : args.resnet50, 
            'opts' : None}
    update_config(cfg, dict_)
    model = get_resnet_model(cfg, is_train=False)
    model.load_state_dict(torch.load(args.path_2d, map_location='cpu'))
    
    return model.eval().cuda()
    


