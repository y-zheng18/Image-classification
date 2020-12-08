import numpy as np
import torch
from torch.nn import DataParallel
from torch.utils.data import DataLoader
from dataset import TrainDataset
import torch.optim.lr_scheduler as lr_scheduler
import torchvision.datasets as datasets
from tqdm import tqdm
from utils.options import get_args
from utils.train_utils import *
from dataset import *
from model.resnet import *
from model.auto_encoder import *


def train(opt):
    if len(opt.gpu_ids) == 1:
        os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpu
    print(opt)
    use_gpu = torch.cuda.is_available()
    print("use_gpu:", use_gpu)
    train_dataset = TrainPairDataset(opt.dataroot, opt.data_type, use_all_data=opt.use_all_data)
    eval_dataset = TrainDataset(opt.dataroot, opt.data_type, phase='eval')
    test_dataset = TestDataset(opt.dataroot)

    train_dataloader = DataLoader(dataset=train_dataset, batch_size=opt.bs, num_workers=0, shuffle=True)
    eval_dataloader = DataLoader(dataset=eval_dataset, batch_size=128, num_workers=0, shuffle=False)
    test_dataloader = DataLoader(dataset=test_dataset, batch_size=128, num_workers=0, shuffle=False)
    test_cifar = DataLoader(
        datasets.CIFAR100(root='./cifar100', train=False, download=True, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
        ])),
        batch_size=128, shuffle=False,
        num_workers=0, pin_memory=True)
    num_classes = 20 if opt.data_type == 'coarse' else 100
    if opt.model == 'resnet':
        backbone_model = ResNet(layers=opt.layers, num_classes=num_classes, dropout_rate=opt.dropout_rate)
    elif opt.model == 'wide_resnet':
        backbone_model = WideResNet(layers=opt.layers, factor=opt.wide_factor, num_classes=num_classes, dropout_rate=opt.dropout_rate)
    elif opt.model == 'multi-res':
        backbone_model = MultiResNet(layers=opt.layers, num_classes=num_classes, dropout_rate=opt.dropout_rate)
    elif opt.model == 'wide_resnext':
        backbone_model = WideResNext(layers=opt.layers, factor=opt.wide_factor, groups=32, num_classes=num_classes)
    else:
        raise NotImplemented

    auto_encoder = AutoEnocder(backbone_model, embedding_size=opt.embedding_size,
                               num_classes=num_classes, fix_backbone=opt.fix_backbone)

    if use_gpu:
        backbone_model.cuda()
        auto_encoder.cuda()

    if opt.optim_policy == 'SGD':
        optimizer = torch.optim.SGD(auto_encoder.parameters(), momentum=0.9, lr=opt.lr, weight_decay=opt.weight_decay)
    else:
        optimizer = torch.optim.Adam(auto_encoder.parameters(), betas=(0.9, 0.95), lr=opt.lr, weight_decay=opt.weight_decay)

    if opt.lr_policy == 'cosine':
        optim_lr_schedule = lr_scheduler.CosineAnnealingLR(optimizer, T_max=opt.epoch)
    else:
        optim_lr_schedule = lr_scheduler.MultiStepLR(optimizer, opt.lr_steps, gamma=opt.lr_decay)
    # optim_lr_schedule = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=opt.lr_tolerance)

    if opt.load_model_dir is not None:
        load(backbone_model, opt.load_model_dir)
    if opt.load_autoencoder_dir is not None:
        load(auto_encoder, opt.load_autoencoder_dir)
    if opt.load_optim_dir is not None:
        load(optimizer, opt.load_optim_dir)
    CLS_loss = nn.CrossEntropyLoss()
    TRI_loss = TripletLoss(margin=opt.triplet_margin)
    REC_loss = nn.MSELoss()

    result_path = opt.result_path
    chk_path = opt.chkpoint_path
    os.makedirs(result_path, exist_ok=True)
    os.makedirs(chk_path, exist_ok=True)

    best_acc = 0
    best_epoch = 0
    backbone_model.eval()
    for epoch in range(opt.epoch_resume, opt.epoch):
        auto_encoder.train()
        loss_list = []
        triplet_loss_list = []
        rec_loss_list = []
        cls_loss_list = []
        train_bar = tqdm(train_dataloader, ncols=120)
        for img_batch, postive_batch, label_batch in train_bar:
            if use_gpu:
                img_batch = img_batch.cuda()
                label_batch = label_batch.cuda()
                postive_batch = postive_batch.cuda()
            optimizer.zero_grad()
            embeddings, ori_feats, reconstruction, predicted = auto_encoder(img_batch)
            embeddings_postive, _, _, _ = auto_encoder(postive_batch)
            triplet_loss = TRI_loss(embeddings, embeddings_postive, label_batch) * opt.lambda_triplet
            classification_loss = CLS_loss(predicted, label_batch) * opt.lambda_cls
            reconstruction_loss = REC_loss(ori_feats, reconstruction) * opt.lambda_rec
            loss = triplet_loss + classification_loss + reconstruction_loss
            loss.backward()
            optimizer.step()

            loss = loss.detach().cpu().numpy()
            triplet_loss = triplet_loss.detach().cpu().numpy()
            classification_loss = classification_loss.detach().cpu().numpy()
            reconstruction_loss = reconstruction_loss.detach().cpu().numpy()

            loss_list.append(loss)
            triplet_loss_list.append(triplet_loss)
            rec_loss_list.append(reconstruction_loss)
            cls_loss_list.append(classification_loss)

            train_bar.set_description('epoch:{0:}, loss:{1:3f}, tri:{2:3f}, cls:{3:3f}, rec:{4:3f}'
                                      .format(epoch, loss, triplet_loss, classification_loss, reconstruction_loss))

        ## evaluating ......
        print('evaluating......')
        eval_acc, eval_acc_cls, anchor_list = eval_metrics(auto_encoder, num_classes, train_dataloader,
                                                           eval_data=eval_dataloader if not opt.use_all_data else test_cifar,
                                                           use_gpu=use_gpu)
        print('eval_acc: {}, eval_cls_acc:{}'.format(eval_acc, eval_acc_cls))
        if best_acc < eval_acc:
            best_acc = eval_acc
            best_epoch = epoch
            save_model(auto_encoder, opt.chkpoint_path,
                       'autoencoder_{}.pth'.format(opt.embedding_size))
            save_model(optimizer, opt.chkpoint_path,
                       'optim_autoencoder_{}.pth'.format(opt.embedding_size))
            test_label_pred = test_metrics(auto_encoder, anchor_list, test_dataloader, use_gpu)
            if opt.data_type == 'coarse':
                save_results(test_label_pred, opt.data_type, opt.result_path,
                             '1_{}_{}.csv'.format(opt.model, "metrics"))
            else:
                save_results(test_label_pred, opt.data_type, opt.result_path,
                             '2_{}_{}.csv'.format(opt.model, "metrics"))
            # if num_classes == 100:
            #     print('cifar100:', test_cifar100(auto_encoder, anchor_list, use_gpu))
        print('epoch:{0:}, lr:{1:6f}, loss:{2:4f}, tri:{3:3f}, cls:{4:3f}, rec:{5:3f}, test_acc:{6:4f}, '
              'best_acc:{7:4f}, best_epoch{8:}'.format(
            epoch, optimizer.param_groups[0]['lr'], np.mean(loss_list), np.mean(triplet_loss_list),
            np.mean(cls_loss_list), np.mean(rec_loss_list),
            eval_acc, best_acc, best_epoch
        ))
        optim_lr_schedule.step()


def eval_metrics(auto_encoder, num_classes, train_data, eval_data, use_gpu=False):
    auto_encoder.eval()
    pred_list = []
    pred_cls_list = []
    gt_list = []
    anchor_list = []
    for i in range(num_classes):
        anchor_list.append([])

    print('extracting anchors')
    train_bar = tqdm(train_data, ncols=100)
    with torch.no_grad():
        for img_batch, postive_batch, label_batch in train_bar:
            if use_gpu:
                img_batch = img_batch.cuda()
                label_batch = label_batch.cuda()
            embeddings, _, _, _ = auto_encoder(img_batch)
            for i, l in enumerate(label_batch):
                anchor_list[l].append((embeddings[i] / torch.norm(embeddings[i])).unsqueeze(0))
        for i, feats in enumerate(anchor_list):
            feats = torch.cat(feats, dim=0)
            cos_distance = torch.mm(feats, feats.permute((1, 0)))
            similarity_sum = torch.sum(cos_distance, dim=1)
            # print(similarity_sum.shape)
            _, max_idx = torch.max(similarity_sum, dim=0)
            anchor_list[i] = feats[max_idx].unsqueeze(0)
        anchor_list = torch.cat(anchor_list, dim=0)
        for img, label in tqdm(eval_data, ncols=100):
            if use_gpu:
                img = img.cuda()
            embeddings, _, _, pred_cls = auto_encoder(img)
            _, predicted_cls_label = torch.max(pred_cls, dim=1)
            pred_cls_list.append(predicted_cls_label.cpu().numpy())
            cos_distance = torch.mm(embeddings, anchor_list.permute((1, 0)))
            _, predicted_label = torch.max(cos_distance, dim=1)
            gt_list.append(label.numpy())
            pred_list.append(predicted_label.cpu().numpy())
        gt_list = np.concatenate(gt_list, axis=0)
        pred_list = np.concatenate(pred_list, axis=0)
        pred_cls_list = np.concatenate(pred_cls_list, axis=0)
        eval_acc = np.sum(pred_list == gt_list) / len(gt_list)
        eval_acc_cls = np.sum(pred_cls_list == gt_list) / len(gt_list)
    return eval_acc, eval_acc_cls, anchor_list


def test_metrics(auto_encoder, anchor_list, test_dataloader, use_gpu):
    auto_encoder.eval()
    pred_list = []

    with torch.no_grad():
        for img in tqdm(test_dataloader, ncols=100):
            if use_gpu:
                img = img.cuda()
            embeddings, _, _, _ = auto_encoder(img)
            cos_distance = torch.mm(embeddings, anchor_list.permute((1, 0)))
            _, predicted_label = torch.max(cos_distance, dim=1)
            pred_list.append(predicted_label.cpu().numpy())
        pred_list = np.concatenate(pred_list, axis=0)
    return pred_list

def test_cifar100(auto_encoder, anchor_list, use_gpu):
    auto_encoder.eval()
    pred_list = []
    gt_list = []
    test_dataloader = DataLoader(
        datasets.CIFAR100(root='./cifar100', train=False, download=True, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
        ])),
        batch_size=128, shuffle=False,
        num_workers=0, pin_memory=True)
    with torch.no_grad():
        for img, label in tqdm(test_dataloader, ncols=100):
            if use_gpu:
                img = img.cuda()
            embeddings, _, _, _ = auto_encoder(img)
            cos_distance = torch.mm(embeddings, anchor_list.permute((1, 0)))
            _, predicted_label = torch.max(cos_distance, dim=1)
            pred_list.append(predicted_label.cpu().numpy())
            gt_list.append(label.numpy())
        gt_list = np.concatenate(gt_list, axis=0)
        pred_list = np.concatenate(pred_list, axis=0)
        test_acc = np.sum(pred_list == gt_list) / len(gt_list)
    return test_acc


if __name__ == "__main__":
    opt = get_args().parse_args()
    train(opt)