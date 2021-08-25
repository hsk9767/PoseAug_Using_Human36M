from __future__ import print_function, absolute_import, division

import os
import os.path as path
import random

import numpy as np
import torch
import torch.backends.cudnn as cudnn

from function_baseline.config import get_parse_args
# from function_baseline.data_preparation import data_preparation
from function_baseline.data_preparation_custom import Data_Custom
from function_baseline.model_pos_preparation import model_pos_preparation
from function_poseaug.model_pos_eval_custom import evaluate, evaluate_2d
from pelee.lib.models.MOBIS_peleenet import get_pose_pelee_net
from common import get_resnet
from data_extra.dataset_converter import COCO2HUMAN, MPII2HUMAN
# from function_poseaug.model_pos_eval import evaluate


def main(args):    
    print('==> Using settings {}'.format(args))
    stride = args.downsample
    cudnn.benchmark = True
    device = torch.device("cuda")

    print('==> Loading dataset...')
    data_class = Data_Custom(is_train=False)
    data_dict = data_class.data_preparation(args)
    print("==> Creating model...")
    model_pos = model_pos_preparation(args, device).cuda()

    # Check if evaluate checkpoint file exist:
    assert path.isfile(args.evaluate), '==> No checkpoint found at {}'.format(args.evaluate)
    print("==> Loading checkpoint '{}'".format(args.evaluate))
    ckpt = torch.load(args.evaluate)
    model_pos.load_state_dict(ckpt['state_dict'])

    print('==> Evaluating...')
    if args.evaluate_2d:
        pck = evaluate_2d(data_dict['valid_loader'], model_pos, device, args.keypoints)
        print('H36M: Protocol #1   (PCK) overall average: {:.2f} (mm)'.format(pck))
    else:
        error_h36m_p1, error_h36m_p2 = evaluate(data_dict['valid_loader'], model_pos, device, args.keypoints, flipaug=False)
        print('H36M: Protocol #1   (MPJPE) overall average: {:.2f} (mm)'.format(error_h36m_p1))
        print('H36M: Protocol #2 (P-MPJPE) overall average: {:.2f} (mm)'.format(error_h36m_p2))

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

    main(args)
