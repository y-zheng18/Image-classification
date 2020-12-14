import numpy as np
from sklearn.manifold import TSNE
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import matplotlib.cm as cm
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import os


def visualize(model, data, save_path, phase='train', metrics=False):
    embedding_list = []
    label_list = []
    model.eval()
    use_gpu = torch.cuda.is_available()
    if use_gpu:
        model = model.cuda()
    with torch.no_grad():
        for img, label in tqdm(data, ncols=100):
            if use_gpu:
                img = img.cuda()
            if metrics:
                embeddings, _, _, _ = model(img)
            else:
                _, embeddings = model(img, return_feats=True)
            embedding_list.append(embeddings.cpu().numpy())
            label_list.append(label.numpy())
        embedding_list = np.concatenate(embedding_list, axis=0)
        label_list = np.concatenate(label_list, axis=0)
    print('fitting embeddings to 2 dim...')
    embeddings_2_dim = TSNE(n_components=2).fit_transform(embedding_list)
    print('fitting done!')
    print(embeddings_2_dim.shape)
    num_classes = np.max(label_list) + 1

    fig = plt.figure(tight_layout=True)
    gs = gridspec.GridSpec(1, 10)

    cmap = cm.Spectral
    norm = Normalize(vmin=0, vmax=np.max(label_list))
    colors = [cmap(norm(i)) for i in label_list]
    label_names = ['{}'.format(i) for i in range(num_classes)]
    pops = [mpatches.Patch(color=cmap(norm(i)), label=label_names[i]) for i in range(num_classes)]
    modelnet = {'dim0': embeddings_2_dim[:, 0], 'dim1': embeddings_2_dim[:, 1], 'y': label_list} # pd.DataFrame({'dim0': data[:, 0], 'dim1': data[:, 1], 'y': label})

    ax = fig.add_subplot(gs[0, 0:9])
    scatter = ax.scatter(
            x=modelnet["dim0"], y=modelnet["dim1"], c=colors, s=10,
            alpha=0.7, edgecolors='none'
            )
    ax.legend(handles=pops, loc='center left', bbox_to_anchor=(1, 0.5), ncol=1, fontsize=7)
    plt.savefig(os.path.join(save_path, phase + 'fig_tsne.png'), dpi=1200)
    plt.show()


def draw_loss(coarse_loss_file, fine_loss_file, save_path):
    coarse = np.load(coarse_loss_file)
    fine = np.load(fine_loss_file)
    #plt.ylim((0, 10000))
    plt.plot(np.arange(200), coarse, color='r', label='loss of coarse prediction')
    plt.plot(np.arange(200), fine, color='b', label='loss of fine prediction')
    plt.legend()
    plt.savefig(os.path.join(save_path, 'loss.png'))
    plt.show()


def draw_acc(coarse_acc_file, fine_acc_file, save_path):
    coarse = np.load(coarse_acc_file)
    fine = np.load(fine_acc_file)
    #plt.ylim((0, 10000))
    plt.plot(np.arange(200), coarse, color='r', label='acc of coarse prediction')
    plt.plot(np.arange(200), fine, color='b', label='acc of fine prediction')
    plt.legend()
    plt.savefig(os.path.join(save_path, 'acc.png'))
    plt.show()



if __name__ == '__main__':
    from utils.options import *
    from utils.train_utils import *
    from model.resnet import *
    from model.auto_encoder import *
    from dataset import *
    import torchvision.models as models

    opt =  get_args().parse_args()
    if len(opt.gpu_ids) == 1:
        os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpu
    print(opt)
    draw_acc('acc_loss/acc_multi-res_coarse.npy', 'acc_loss/acc_multi-res_fine.npy', opt.result_path)
    draw_loss('acc_loss/loss_multi-res_coarse.npy', 'acc_loss/loss_multi-res_fine.npy', opt.result_path)

    use_gpu = torch.cuda.is_available()
    print("use_gpu:", use_gpu)
    train_dataset = TrainDataset(opt.dataroot, opt.data_type, phase='train' if not opt.use_all_data else 'all', pretrained=opt.pretrained)
    eval_dataset = TrainDataset(opt.dataroot, opt.data_type, phase='eval', pretrained=opt.pretrained)

    train_dataloader = DataLoader(dataset=train_dataset, batch_size=opt.bs, num_workers=0, shuffle=True)
    eval_dataloader = DataLoader(dataset=eval_dataset, batch_size=128, num_workers=0, shuffle=False)

    num_classes = 20 if opt.data_type == 'coarse' else 100
    if opt.model == 'resnet':
        model = ResNet(layers=opt.layers, num_classes=num_classes, dropout_rate=opt.dropout_rate)
    elif opt.model == 'wide_resnet':
        model = WideResNet(layers=opt.layers, factor=opt.wide_factor, num_classes=num_classes, dropout_rate=opt.dropout_rate)
    elif opt.model == 'multi-res':
        model = MultiResNet(layers=opt.layers, num_classes=num_classes, dropout_rate=opt.dropout_rate)
    elif opt.model == 'wide_resnext':
        model = WideResNext(layers=opt.layers, factor=opt.wide_factor, groups=32, num_classes=num_classes)
    elif opt.model == 'resnet_pretrained':
        model = models.resnet152()
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif opt.model == 'wideresnet_pretrained':
        model = models.wide_resnet101_2()
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    else:
        raise NotImplemented
    if opt.use_triplet:
        model = AutoEnocder(model, embedding_size=opt.embedding_size,
                            num_classes=num_classes, fix_backbone=opt.fix_backbone)
        load(model, opt.load_autoencoder_dir)
    else:
        load(model, opt.load_model_dir)
    visualize(model, train_dataloader, phase='train', save_path=opt.result_path, metrics=opt.use_triplet)