import argparse
import os

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=str, default='2')

    parser.add_argument('--bs', type=int, default=128)
    parser.add_argument('--dataroot', type=str, default='dataset/')
    parser.add_argument('--data_type', type=str, default='coarse', choices=['coarse', 'fine'])
    parser.add_argument('--chkpoint_path', type=str, default='chkpoints/')
    parser.add_argument('--result_path', type=str, default='results/')
    parser.add_argument('--load_model_dir', type=str, default=None)
    parser.add_argument('--load_optim_dir', type=str, default=None)

    parser.add_argument('--model', type=str, default='resnet', choices=['resnet', 'wide_resnet'])
    parser.add_argument('--epoch', type=int, default=200)
    parser.add_argument('--epoch_resume', type=int, default=0)
    parser.add_argument('--lr', type=float, default=1e-1)
    parser.add_argument('--lr_policy', type=str, default='cosine', choices=['cosine', 'multi-step'])
    parser.add_argument('--lr_steps', type=int, default=[10, 40, 80], nargs='+')
    parser.add_argument('--lr_decay', type=float, default=0.2)
    parser.add_argument('--lr_tolerance', type=int, default=1)
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    parser.add_argument('--dropout_rate', type=float, default=0.3)
    parser.add_argument('--save_fre', type=int, default=10)

    parser.add_argument('--use_triplet', default=False, action='store_true')
    parser.add_argument('--triplet_margin', type=float, default=0.3)
    parser.add_argument('--triplet_warm_up', type=int, default=5)
    parser.add_argument('--lambda_triplet', type=float, default=1)
    return parser
