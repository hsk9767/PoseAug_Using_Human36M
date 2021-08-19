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
from function_poseaug.model_pos_eval_custom import evaluate, evaluate_3d_mppe
from pelee.lib.models.MOBIS_peleenet import get_pose_pelee_net
from common import get_resnet
from data_extra.dataset_converter import COCO2HUMAN, MPII2HUMAN
from one_stage import get_pose_net
# from function_poseaug.model_pos_eval import evaluate
from common.common_dataset import DatasetLoader_3d_mppe
import torchvision.transforms as transforms
from torch.utils.data import DataLoader


pixel_mean = (0.485, 0.456, 0.406)
pixel_std = (0.229, 0.224, 0.225)

def main(args):    
    print('==> Using settings {}'.format(args))
    cudnn.benchmark = True
    device = torch.device("cuda")

    print('==> Loading dataset...')
    path_3d = 'common.' + 'h36m_dataset_custom'
    exec('from ' + path_3d + ' import ' + 'Human36M')
    dataset_3d = DatasetLoader_3d_mppe(eval('Human36M')('test', True), ref_joints_name=None, is_train=False, transform=transforms.Compose([\
                                                                                                                        transforms.ToTensor()
                                                                                                                        , transforms.Normalize(mean=pixel_mean, std=pixel_std)]))
    valid_loader = DataLoader(dataset_3d, batch_size=32, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    
    print("==> Creating model...")
    model = get_pose_net(50, False, 18)
    model = torch.nn.DataParallel(model)
    model.load_state_dict(torch.load(args.path_one_stage, map_location='cpu')['network'])
    print(f"==> Loading from {args.path_one_stage}")
    model = model.cuda().eval()

    print('==> Evaluating...')
    
    # error_h36m_p1, error_h36m_p2 = evaluate(valid_loader, model, device, args.keypoints, flipaug=False)
    evaluate_3d_mppe(valid_loader, model, device)

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
