import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from dataset import TrainDataset
import torch.optim.lr_scheduler as lr_scheduler
from tqdm import tqdm
from utils.options import get_args
from utils.train_utils import *
from dataset import *
from model.resnet import *
from torchvision import models
import torchvision.datasets as datasets


def test_metrics(opt):
    os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpu
    print(opt)
    use_gpu = torch.cuda.is_available()
    print("use_gpu:", use_gpu)

    test_dataloader = DataLoader(
        datasets.CIFAR100(root='./cifar100', train=False, download=True, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
        ])),
        batch_size=128, shuffle=False,
        num_workers=0, pin_memory=True)
    num_classes = 100
    if opt.model == 'resnet':
        model = ResNet(layers=opt.layers, num_classes=num_classes,
                       dropout_rate=opt.dropout_rate)
    elif opt.model == 'wide_resnet':
        model = WideResNet(layers=opt.layers, factor=opt.wide_factor, num_classes=num_classes,
                           dropout_rate=opt.dropout_rate)
    elif opt.model == 'multi-res':
        model = MultiResNet(layers=opt.layers, num_classes=num_classes, dropout_rate=opt.dropout_rate)
    else:
        model = ResNetMetrics(layers=opt.layers, num_classes=num_classes, dropout_rate=opt.dropout_rate)
    if use_gpu:
        model.cuda()

    if opt.load_model_dir is not None:
        print('loading model from {}'.format(opt.load_model_dir))
        load(model, opt.load_model_dir)
        print('load done!')

    result_path = opt.result_path
    os.makedirs(result_path, exist_ok=True)
    model.eval()
    with torch.no_grad():
        pred_list = []
        gt_list = []
        print('evaluating')
        test_bar = tqdm(test_dataloader, ncols=100)
        for img_batch, label_batch in test_bar:
            if use_gpu:
                img_batch = img_batch.cuda()
                # postive_batch = postive_batch.cuda()
            predicted = model(img_batch)
            _, predicted_label = torch.max(predicted, dim=1)
            pred_list.append(predicted_label.cpu().numpy())
            gt_list.append(label_batch.numpy())
        gt_list = np.concatenate(gt_list, axis=0)
        pred_list = np.concatenate(pred_list, axis=0)
        eval_acc = np.sum(pred_list == gt_list) / len(gt_list)
        print("cifar100 accuracy:", eval_acc)

if __name__ == "__main__":
    opt = get_args().parse_args()
    test_metrics(opt)