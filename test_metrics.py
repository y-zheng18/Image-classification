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


def test_metrics(opt):
    os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpu
    print(opt)
    use_gpu = torch.cuda.is_available()
    print("use_gpu:", use_gpu)
    train_dataset = TrainPairDataset(opt.dataroot, opt.data_type)
    eval_dataset = TrainDataset(opt.dataroot, opt.data_type, phase='eval')
    test_dataset = TestDataset(opt.dataroot)

    train_dataloader = DataLoader(dataset=train_dataset, batch_size=opt.bs, num_workers=0, shuffle=False)
    eval_dataloader = DataLoader(dataset=eval_dataset, batch_size=128, num_workers=0, shuffle=False)
    test_dataloader = DataLoader(dataset=test_dataset, batch_size=128, num_workers=0, shuffle=False)
    num_classes = 20 if opt.data_type == 'coarse' else 100
    model = ResNetMetrics(layers=(2, 2, 2, 2), num_classes=num_classes, dropout_rate=opt.dropout_rate) \
        if opt.model == 'resnet' else WideResNetMetrics(layers=opt.layers, factor=opt.wide_factor,
                                                        num_classes=num_classes, dropout_rate=opt.dropout_rate)

    if use_gpu:
        model.cuda()

    if opt.load_model_dir is not None:
        print('loading model from {}'.format(opt.load_model_dir))
        load(model, opt.load_model_dir)
        print('load done!')

    result_path = opt.result_path
    os.makedirs(result_path, exist_ok=True)
    discrimitive = TripletLoss()
    best_acc = 0
    best_epoch = 0
    model.eval()
    with torch.no_grad():
        pred_list = []
        gt_list = []
        anchor_list = []
        for i in range(num_classes):
            anchor_list.append([])
        print('extracting anchors')
        train_bar = tqdm(train_dataloader, ncols=100)
        for img_batch, postive_batch, label_batch in train_bar:
            if use_gpu:
                img_batch = img_batch.cuda()
                label_batch = label_batch.cuda()
            _, embeddings = model(img_batch, return_feats=True)
            for i, l in enumerate(label_batch):
                # print(embeddings[i].shape)
                anchor_list[l].append((embeddings[i] / torch.norm(embeddings[i])).unsqueeze(0))
        for i, feats in enumerate(anchor_list):
            feats = torch.cat(feats, dim=0)
            cos_distance = torch.mm(feats, feats.permute((1, 0)))
            similarity_sum = torch.sum(cos_distance, dim=1)
            # print(similarity_sum.shape)
            _, max_idx = torch.max(similarity_sum, dim=0)
            anchor_list[i] = feats[0].unsqueeze(0)
        anchor_list = torch.cat(anchor_list, dim=0)
        print('evaluating......')
        for img, label in tqdm(eval_dataloader, ncols=100):
            if use_gpu:
                img = img.cuda()
            _, embeddings = model(img, return_feats=True)
            # print(embeddings.shape, anchor_list.permute((1, 0)).shape)
            cos_distance = torch.mm(embeddings, anchor_list.permute((1, 0)))
            _, predicted_label = torch.max(cos_distance, dim=1)
            gt_list.append(label.numpy())
            pred_list.append(predicted_label.cpu().numpy())

        gt_list = np.concatenate(gt_list, axis=0)
        pred_list = np.concatenate(pred_list, axis=0)
        eval_acc = np.sum(pred_list == gt_list) / len(gt_list)
        print(eval_acc)

if __name__ == "__main__":
    opt = get_args().parse_args()
    test_metrics(opt)