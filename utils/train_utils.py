import numpy as np
import torch
import os
import pandas as pd
from tqdm import tqdm
import torch.nn.functional as F


def eval(model, eval_data, use_gpu=False):
    model.eval()
    with torch.no_grad():
        pred_list = []
        gt_list = []
        for img, label in tqdm(eval_data, ncols=100):
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

def test(model, test_data,  use_gpu=False):
    model.eval()
    with torch.no_grad():
        pred_list = []
        for img in tqdm(test_data, ncols=100):
            if use_gpu:
                img = img.cuda()
            predicted = model(img)
            _, predicted_label = torch.max(predicted, dim=1)
            pred_list.append(predicted_label.cpu().numpy())
        pred_list = np.concatenate(pred_list)
    return pred_list


def load(model, chkpoints_path):
    print('loading for net ...', chkpoints_path)
    pretrained_dict = torch.load(chkpoints_path)
    model.load_state_dict(pretrained_dict)
    print("loaded finished!")


def save_model(model, save_root, save_name):
    os.makedirs(save_root, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(save_root, save_name))


def save_results(pred_label, data_type, save_root, save_name):
    os.makedirs(save_root, exist_ok=True)
    data = np.array([np.arange(len(pred_label)), pred_label]).T
    data = pd.DataFrame(data, columns=["image_id", data_type])
    data.to_csv(os.path.join(save_root, save_name), index=False)


class TripletLoss(torch.nn.Module):
    def __init__(self, margin=0.3):
        super(TripletLoss, self).__init__()
        self.margin = margin

    def forward(self, anchor_feats, postive_feats, labels):
        """
            Inputs:
            - sp: similarity between postive samples, shape (batchsize)
            - sn: similarity between negative samples, shape (batchsize)
            Output:
            - triplet loss
        """
        negative_feats = self.get_nagetive(anchor_feats, labels)
        sn = self.Cosine(anchor_feats, negative_feats)
        sp = self.Cosine(anchor_feats, postive_feats)
        loss = torch.mean(torch.clamp(sn - sp + self.margin, min=0))
        return loss

    def get_nagetive(self, features, labels):
        bs = features.shape[0]
        normal_anchor = F.normalize(features, dim=1)  # (bs, dim)
        adjacent = torch.matmul(normal_anchor, normal_anchor.transpose(1, 0))  # (batch_size, batch_size)
        # anchor_index = [i for i in range(bs)]
        # anchor_adjacent = adjacent[anchor_index]  # (batch_size, 2 * batch_size)
        for i in range(bs):
            # anchor_adjacent[i, i] = -1
            pair_index = (labels == labels[i])
            adjacent[i, pair_index] = -1
        negative_index = torch.argmax(adjacent, dim=1)  # hardest negative sample
        negative_feats = features[negative_index]
        return negative_feats

    def Cosine(self, b1, b2):
        """
            Inputs:
                - features1: shape (batchsize, dim)
                - features2: shape (batchsize, dim)
            Output:
                - cosine distance between features1 and features2
        """
        normal1 = F.normalize(b1, dim=1)
        normal2 = F.normalize(b2, dim=1)
        batch_size = normal1.shape[0]
        cos = torch.sum(normal1 * normal2, 1)
        cos = cos.reshape(batch_size)
        return cos
# save_results(np.arange(100), 'coarse_label', '../results', '1.csv')