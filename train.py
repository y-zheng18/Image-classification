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



def train():
    opt = get_args().parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpu
    print(opt)
    use_gpu = torch.cuda.is_available()
    print("use_gpu:", use_gpu)
    train_dataset = TrainDataset(opt.dataroot, opt.data_type, phase='train')
    eval_dataset = TrainDataset(opt.dataroot, opt.data_type, phase='eval')
    test_dataset = TestDataset(opt.dataroot)

    train_dataloader = DataLoader(dataset=train_dataset, batch_size=opt.bs, num_workers=0, shuffle=True)
    eval_dataloader = DataLoader(dataset=eval_dataset, batch_size=128, num_workers=0, shuffle=False)
    test_dataloader = DataLoader(dataset=test_dataset, batch_size=128, num_workers=0, shuffle=False)

    if opt.model == 'resnet':
        model = ResNet(layers=(2, 2, 2, 2), num_classes=20 if opt.data_type == 'coarse' else 200, dropout_rate=opt.dropout_rate)
    else:
        model = WideResNet(layers=(4, 4, 4), num_classes=20 if opt.data_type == 'coarse' else 200, dropout_rate=opt.dropout_rate)
    #model = models.resnet50(num_classes=20)
    if use_gpu:
        model.cuda()
    optimizer = torch.optim.SGD(model.parameters(), momentum=0.9, lr=opt.lr, weight_decay=opt.weight_decay)
    if opt.lr_policy == 'cosine':
        optim_lr_schedule = lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)
    else:
        optim_lr_schedule = lr_scheduler.MultiStepLR(optimizer, opt.lr_steps, gamma=opt.lr_decay)
    # optim_lr_schedule = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=opt.lr_tolerance)

    if opt.load_model_dir is not None:
        load(model, opt.load_model_dir)
    if opt.load_optim_dir is not None:
        load(optimizer, opt.load_optim_dir)
    loss_function = nn.CrossEntropyLoss()

    result_path = opt.result_path
    chk_path = opt.chkpoint_path
    os.makedirs(result_path, exist_ok=True)
    os.makedirs(chk_path, exist_ok=True)

    best_acc = 0
    best_epoch = 0
    for epoch in range(opt.epoch_resume, opt.epoch):
        model.train()
        loss_list = []
        pred_list = []
        gt_list = []
        # it = 0
        for img_batch, label_batch in tqdm(train_dataloader, ncols=100):
            if use_gpu:
                img_batch = img_batch.cuda()
                label_batch = label_batch.cuda()
            optimizer.zero_grad()
            outputs = model(img_batch)
            loss = loss_function(outputs, label_batch)
            loss.backward()
            optimizer.step()
            loss_list.append(loss.detach().cpu().numpy())
            # if it % 10 == 0:
            #     print('epoch:{0:}, lr:{1:06f}, it:{2:}, loss:{3:04f}'.format(
            #         epoch, optimizer.param_groups[0]['lr'], it, loss))
            # it += 1

            _, predicted_label = torch.max(outputs, dim=1)
            gt_list.append(label_batch.cpu().numpy())
            pred_list.append(predicted_label.cpu().numpy())
        gt_list = np.concatenate(gt_list, axis=0)
        pred_list = np.concatenate(pred_list, axis=0)
        train_acc = np.sum(pred_list == gt_list) / len(gt_list)
        acc = eval(model, eval_dataloader, use_gpu)
        if best_acc < acc:
            best_acc = acc
            best_epoch = epoch
            save_model(model, opt.chkpoint_path, '{}_{}.pth'.format(opt.model, opt.data_type))
            save_model(optimizer, opt.chkpoint_path, 'optim_{}_{}.pth'.format(opt.model, opt.data_type))
            test_label_pred = test(model, test_dataloader, use_gpu)
            if opt.data_type == 'coarse':
                save_results(test_label_pred, opt.data_type, opt.result_path, '1_{}.csv'.format(opt.model))
            else:
                save_results(test_label_pred, opt.data_type, opt.result_path, '2_{}.csv'.format(opt.model))
        print('epoch:{0:}, lr:{1:6f}, loss:{2:4f}, train_acc:{3:4f}, test_acc:{4:4f}, best_acc:{5:4f}, best_epoch{6:}'.format(
            epoch, optimizer.param_groups[0]['lr'], np.mean(loss_list), train_acc, acc, best_acc, best_epoch
        ))
        # loss_list = np.array(loss_list)
        optim_lr_schedule.step()


if __name__ == "__main__":
    train()