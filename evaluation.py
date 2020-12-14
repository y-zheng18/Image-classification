from utils.options import *
from utils.train_utils import *
from model.resnet import *
from model.auto_encoder import *
from dataset import *
import torchvision.models as models
import torchvision.datasets as datasets

def eval(model, datasets):
    model.eval()
    print('evaluating......')
    use_gpu = torch.cuda.is_available()
    if use_gpu:
        model = model.cuda()
    with torch.no_grad():
        t1_correct_num = 0
        t5_correct_num = 0
        total_num = 0
        for img, label in tqdm(datasets, ncols=100):
            if use_gpu:
                img = img.cuda()
            total_num += img.shape[0]
            predicted = model(img)
            _, predicted_label = torch.topk(predicted, 3, dim=1)
            predicted_label = predicted_label.cpu().numpy()
            label = label.numpy()
            t1_correct_num += np.sum(predicted_label[:, 0] == label)
            for i in range(img.shape[0]):
                t5_correct_num += label[i] in predicted_label[i]
        t1_acc = t1_correct_num / total_num
        t5_acc = t5_correct_num / total_num
    print('t1 acc:{}, t5 acc:{}'.format(t1_acc, t5_acc))
    return t1_acc, t5_acc




if __name__ == '__main__':
    opt =  get_args().parse_args()
    if len(opt.gpu_ids) == 1:
        os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpu
    print(opt)

    use_gpu = torch.cuda.is_available()
    print("use_gpu:", use_gpu)
    eval_dataset = TrainDataset(opt.dataroot, opt.data_type, phase='eval', pretrained=opt.pretrained)

    eval_dataloader = DataLoader(dataset=eval_dataset, batch_size=128, num_workers=0, shuffle=False)

    num_classes = 20 if opt.data_type == 'coarse' else 100
    if opt.model == 'resnet':
        model = ResNet(layers=opt.layers, num_classes=num_classes, dropout_rate=opt.dropout_rate)
    elif opt.model == 'wide_resnet':
        model = WideResNet(layers=opt.layers, factor=opt.wide_factor, num_classes=num_classes, dropout_rate=opt.dropout_rate)
    elif opt.model == 'multi-res':
        model = MultiWideResNet(layers=opt.layers, num_classes=num_classes, dropout_rate=opt.dropout_rate)
    elif opt.model == 'resnet_pretrained':
        model = models.resnet152()
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif opt.model == 'wideresnet_pretrained':
        model = models.wide_resnet101_2()
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    else:
        raise NotImplemented

    load(model, opt.load_model_dir)
    if opt.data_type == 'coarse':
        eval(model, eval_dataloader)
    else:
        size = 32 if not opt.pretrained else 224
        test_dataloader = DataLoader(
            datasets.CIFAR100(root='./cifar100', train=False, download=True, transform=transforms.Compose([
                transforms.Resize((size, size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
            ])),
            batch_size=128, shuffle=False,
            num_workers=0, pin_memory=True)
        eval(model, test_dataloader)