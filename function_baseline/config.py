import argparse



def get_parse_args():
    parser = argparse.ArgumentParser(description='PyTorch training script')

    # General arguments
    parser.add_argument('--dataset', default='h36m', type=str, metavar='NAME', help='target dataset')
    parser.add_argument('--keypoints', default='gt', type=str, metavar='NAME', help='2D detections to use, \
    gt/hr/cpn_ft_h36m_dbb/detectron_ft_h36m')
    parser.add_argument('--path_2d', default = '../final_state.pth.tar', type=str, help = 'weight path of the 2D estimation network')
    parser.add_argument('--pelee_imagenet_pretrain_path', default = 'data/Human3.6M/peleenet_acc7208.pth.tar', type=str, help = 'peleenet weight to finetune')
    parser.add_argument('--actions', default='*', type=str, metavar='LIST',
                        help='actions to train/test on, separated by comma, or * for all')
    parser.add_argument('--checkpoint', default='checkpoint/debug', type=str, metavar='PATH',
                        help='checkpoint directory')
    parser.add_argument('--snapshot', default=25, type=int, help='save models_baseline for every (default: 20)')
    parser.add_argument('--note', default='debug', type=str, help='additional name on checkpoint directory')
    parser.add_argument('--resnet50', default='hrnet_simplebaseline/experiments/mpii/resnet/res50_256x256_d256x3_adam_lr1e-3.yaml', 
                        type=str, help='where the cfg of the model(simplebaseline-resnet50) is taken from')
    parser.add_argument('--hrnet32', default='hrnet_simplebaseline/experiments/mpii/hrnet/w32_256x256_adam_lr1e-3.yaml', 
                        type=str, help='where the cfg of the model(hrnet-32) is taken from')
    parser.add_argument('--saved_2d', default='resnet.npz', type=str, help='results of 2D pose estimation network about Human3.6M')
    parser.add_argument('--is_train', default=True, type=lambda x: (str(x).lower() == 'true'), help='train_or_valid')
    parser.add_argument('--vis_save_path', default='data/Human3.6M/viz/', type=str, help='results of 2D pose estimation network about Human3.6M')
    parser.add_argument('--what_to_vis', default='s_09_act_06_subact_01_ca_01', type=str, help='results of 2D pose estimation network about Human3.6M')
    
    # one-stage 
    parser.add_argument('--one_stage_continue_train', default=False, type=lambda x: (str(x).lower() == 'true'), help='continue one-stage model train?')
    parser.add_argument('--path_one_stage', default = 'data/Human3.6M/snapshot_24.pth.tar', type=str, help = 'weight path of the one_stage method')
    parser.add_argument('--save_path_one_stage', default = 'data/Human3.6M/one_stage/', type=str, help = 'save path of the one_stage method')
    parser.add_argument('--dec_start', default=17, type=int, metavar='N', help='the first epoch to lr_decay')
    parser.add_argument('--dec_end', default=21, type=int, metavar='N', help='the last epoch to lr_decay')
    parser.add_argument('--dec_fac', default=10, type=int, metavar='N', help='the last epoch to lr_decay')
    
    
    # Evaluate choice
    parser.add_argument('--evaluate', default='', type=str, metavar='FILENAME',
                        help='checkpoint to evaluate (file name)')
    parser.add_argument('--action-wise', default=True, type=lambda x: (str(x).lower() == 'true'), help='train s1only')

    # Model arguments
    parser.add_argument('--posenet_name', default='videopose', type=str, help='posenet: gcn/stgcn/videopose/mlp')
    parser.add_argument('--stages', default=4, type=int, metavar='N', help='stages of baseline model')
    parser.add_argument('--dropout', default=0.25, type=float, help='dropout rate')

    # Training detail
    parser.add_argument('--batch_size', default=64, type=int, metavar='N',
                        help='batch size in terms of predicted frames')
    parser.add_argument('--epochs', default=200, type=int, metavar='N', help='number of training epochs')

    # Learning rate
    parser.add_argument('--lr', default=1.0e-3, type=float, metavar='LR', help='initial learning rate')
    parser.add_argument('--lr_decay', type=int, default=3000, help='num of steps of learning rate decay')
    parser.add_argument('--lr_gamma', type=float, default=0.96, help='gamma of learning rate decay')
    parser.add_argument('--no_max', dest='max_norm', action='store_false', help='if use max_norm clip on grad')
    parser.set_defaults(max_norm=True)

    # Experimental setting
    parser.add_argument('--random_seed', type=int, default=0)
    parser.add_argument('--downsample', default=1, type=int, metavar='FACTOR', help='downsample frame rate by factor')
    parser.add_argument('--pretrain', default=False, type=lambda x: (str(x).lower() == 'true'), help='used in poseaug')
    parser.add_argument('--s1only', default=False, type=lambda x: (str(x).lower() == 'true'), help='train S1 only')
    parser.add_argument('--num_workers', default=2, type=int, metavar='N', help='num of workers for data loading')

    args = parser.parse_args()

    return args

