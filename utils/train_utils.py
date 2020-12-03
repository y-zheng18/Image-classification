import numpy as np
import torch
import os
import pandas as pd
from tqdm import tqdm

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

# save_results(np.arange(100), 'coarse_label', '../results', '1.csv')