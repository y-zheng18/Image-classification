import numpy as np
import torch
from torch.nn import DataParallel
from torch.utils.data import DataLoader
from dataset import TrainDataset
import torch.optim.lr_scheduler as lr_scheduler
from tqdm import tqdm
import torchvision.models as models
from utils.options import get_args
from utils.train_utils import *
from dataset import *
from model.resnet import *
import torchvision.datasets as datasets


def train(opt):
    if len(opt.gpu_ids) == 1:
        os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpu
    print(opt)
    use_gpu = torch.cuda.is_available()
    print("use_gpu:", use_gpu)
    train_dataset = TrainDataset(opt.dataroot, opt.data_type, phase='train' if not opt.use_all_data else 'all', pretrained=opt.pretrained)
    eval_dataset = TrainDataset(opt.dataroot, opt.data_type, phase='eval', pretrained=opt.pretrained)
    test_dataset = TestDataset(opt.dataroot, pretrained=opt.pretrained)

    train_dataloader = DataLoader(dataset=train_dataset, batch_size=opt.bs, num_workers=0, shuffle=True)
    eval_dataloader = DataLoader(dataset=eval_dataset, batch_size=128, num_workers=0, shuffle=False)
    test_dataloader = DataLoader(dataset=test_dataset, batch_size=128, num_workers=0, shuffle=False)

    if opt.model == 'resnet':
        model = ResNet(layers=opt.layers, num_classes=20 if opt.data_type == 'coarse' else 100, dropout_rate=opt.dropout_rate)
    elif opt.model == 'wide_resnet':
        model = WideResNet(layers=opt.layers, factor=opt.wide_factor, num_classes=20 if opt.data_type == 'coarse' else 100, dropout_rate=opt.dropout_rate)
    elif opt.model == 'multi-res':
        model = MultiResNet(layers=opt.layers, num_classes=20 if opt.data_type == 'coarse' else 100, dropout_rate=opt.dropout_rate)
    elif opt.model == 'wide_resnext':
        model = WideResNext(layers=opt.layers, factor=opt.wide_factor, groups=32, num_classes=20 if opt.data_type == 'coarse' else 100)
    elif opt.model == 'resnet_pretrained':
        model = models.resnet152(pretrained=True)
        model.fc = nn.Linear(model.fc.in_features, 20 if opt.data_type == 'coarse' else 100)
    elif opt.model == 'wideresnet_pretrained':
        model = models.wide_resnet50_2(pretrained=True)
        model.fc = nn.Linear(model.fc.in_features, 20 if opt.data_type == 'coarse' else 100)
    else:
        raise NotImplemented
    # print(model)
    if use_gpu:
        model.cuda()
    if len(opt.gpu_ids) > 1:
        print("gpu_ids:{}".format(opt.gpu_ids))
        model = DataParallel(model, device_ids=opt.gpu_ids)
    if opt.optim_policy == 'SGD':
        optimizer = torch.optim.SGD(model.parameters(), momentum=0.9, lr=opt.lr, weight_decay=opt.weight_decay)
    else:
        optimizer = torch.optim.Adam(model.parameters(), betas=(0.9, 0.95), lr=opt.lr, weight_decay=opt.weight_decay)

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
    loss_list = []
    for epoch in range(opt.epoch_resume, opt.epoch):
        model.train()
        loss_list = []
        pred_list = []
        gt_list = []
        # it = 0
        train_bar = tqdm(train_dataloader, ncols=100)
        for img_batch, label_batch in train_bar:
            if use_gpu:
                img_batch = img_batch.cuda()
                label_batch = label_batch.cuda()
            optimizer.zero_grad()
            outputs = model(img_batch)
            loss = loss_function(outputs, label_batch)
            if len(opt.gpu_ids) > 1:
                loss = loss.mean()
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
            train_bar.set_description('epoch:{0:}, loss:{1:4f}'.format(epoch, loss))
        gt_list = np.concatenate(gt_list, axis=0)
        pred_list = np.concatenate(pred_list, axis=0)
        train_acc = np.sum(pred_list == gt_list) / len(gt_list)
        if len(opt.gpu_ids) > 1:
            model = model.module
        if opt.use_all_data and opt.data_type == 'fine':
            acc = test_cifar100(model, eval_dataloader, use_gpu)
        else:
            acc = eval(model, eval_dataloader, use_gpu)
        if best_acc < acc:
            best_acc = acc
            best_epoch = epoch
            save_model(model, opt.chkpoint_path,
                       '{}_{}_{}.pth'.format(opt.model, opt.optim_policy, opt.data_type))
            save_model(optimizer, opt.chkpoint_path,
                       'optim_{}_{}_{}.pth'.format(opt.model, opt.optim_policy, opt.data_type))
            test_label_pred = test(model, test_dataloader, use_gpu)
            if opt.data_type == 'coarse':
                save_results(test_label_pred, opt.data_type, opt.result_path,
                             '1_{}_{}.csv'.format(opt.model, opt.optim_policy))
            else:
                save_results(test_label_pred, opt.data_type, opt.result_path,
                             '2_{}_{}.csv'.format(opt.model, opt.optim_policy))
        print('epoch:{0:}, lr:{1:6f}, loss:{2:4f}, train_acc:{3:4f}, test_acc:{4:4f}, best_acc:{5:4f}, best_epoch{6:}'.format(
            epoch, optimizer.param_groups[0]['lr'], np.mean(loss_list), train_acc, acc, best_acc, best_epoch
        ))
        # loss_list = np.array(loss_list)
        if len(opt.gpu_ids) > 1:
            model = DataParallel(model, device_ids=opt.gpu_ids)
        optim_lr_schedule.step()
        loss_list.append(np.mean(loss_list))
    np.save(os.path.join(result_path, 'loss_{}_{}.npy'.format(opt.model, opt.data_type)), np.array(loss_list))

def test_cifar100(model, eval_data, use_gpu):
    model.eval()
    pred_list = []
    gt_list = []
    size = eval_data.dataset.size
    test_dataloader = DataLoader(
        datasets.CIFAR100(root='./cifar100', train=False, download=True, transform=transforms.Compose([
            transforms.Resize((size, size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
        ])),
        batch_size=128, shuffle=False,
        num_workers=0, pin_memory=True)
    with torch.no_grad():
        for img, label in tqdm(test_dataloader, ncols=100):
            if use_gpu:
                img = img.cuda()
            predicted = model(img)
            _, predicted_label = torch.max(predicted, dim=1)
            gt_list.append(label)
            pred_list.append(predicted_label.cpu().numpy())
        gt_list = np.concatenate(gt_list, axis=0)
        pred_list = np.concatenate(pred_list, axis=0)
        acc = np.sum(pred_list == gt_list) / len(gt_list)
    return acc


if __name__ == "__main__":
    opt = get_args().parse_args()
    train(opt)