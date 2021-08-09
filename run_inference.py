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
# from function_baseline.data_preparation import data_preparation
from function_baseline.model_pos_preparation import model_pos_preparation
# from function_baseline.model_pos_train import train
# from function_poseaug.model_pos_eval import evaluate
# from utils.log import Logger, savefig
# from utils.utils import save_ckpt

"""
this code is used to pretrain the baseline model
1. Simple Baseline
2. VideoPose
3. SemGCN
4. ST-GCN
code are modified from https://github.com/garyzhao/SemGCN
"""
import time

def main(args):
    print('==> Using settings {}'.format(args))
    device = torch.device("cuda")

    print('==> Loading dataset...')
    # data_dict = data_preparation(args)

    print("==> Creating PoseNet model...")
    # model_pos = model_pos_preparation(args, data_dict['dataset'], device)
    model_pos = model_pos_preparation(args, device).eval()
    
    dummy_input = torch.rand(size=(10, 16, 2)).cuda()
    
    for i in range(500):
        model_pos(dummy_input)
    
    start_time = time.time()
    for i in range(500):
        model_pos(dummy_input)
        torch.cuda.synchronize()
    print('time : ', (time.time() - start_time)/500)
    
    



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
