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
from function_poseaug.model_pos_eval_custom import evaluate, evaluate_2d, evaluate_only_2d
from pelee.lib.models.MOBIS_peleenet import get_pose_pelee_net
from common import get_resnet
from data_extra.dataset_converter import COCO2HUMAN, MPII2HUMAN
from common.common_dataset import DatasetLoader
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from one_stage import get_pose_net
# from function_poseaug.model_pos_eval import evaluate

pixel_mean = (0.485, 0.456, 0.406)
pixel_std = (0.229, 0.224, 0.225)
def main(args):   
    print('==> Using settings {}'.format(args))
    stride = args.downsample
    cudnn.benchmark = True
    device = torch.device("cuda")

    path_3d = 'common.' + 'h36m_dataset_custom'
    exec('from ' + path_3d + ' import ' + 'Human36M')
    
    print('==> Loading dataset...')
    dataset = DatasetLoader(eval('Human36M')('test'), ref_joints_name=None, is_train=False, transform=transforms.Compose([\
                                                                            transforms.ToTensor()
                                                                            , transforms.Normalize(mean=pixel_mean, std=pixel_std)]), detection_2d = True, only_2d = True)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    print("==> Creating model...")
    if args.keypoints == 'pelee':
        estimator = get_pose_pelee_net(False).cuda().eval()
        estimator.load_state_dict(torch.load(args.path_2d, map_location='cpu'))
    elif 'resnet' in args.keypoints:
        estimator = get_resnet(args).cuda()
        estimator.load_state_dict(torch.load(args.path_2d, map_location='cpu'))
    elif 'one_stage' == args.keypoints:
        estimator = get_pose_net(50, False, 18).cuda()
        estimator = torch.nn.DataParallel(estimator)
        estimator.load_state_dict(torch.load(args.path_one_stage, map_location='cpu')['network'])
    else:
        raise NotImplementedError("Not supported")
    estimator.eval()

    print('==> Evaluating...')
    assert args.only_2d
    pck = evaluate_only_2d(loader, estimator, device, args.keypoints)
    print('H36M: Protocol #1   (PCK) overall average: {:.2f} (mm)'.format(pck))

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
