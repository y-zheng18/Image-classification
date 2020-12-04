import torch
from torchvision import transforms
import os
from torch.utils.data import Dataset, DataLoader
import cv2
import pandas as pd
import numpy as np
from PIL import Image


class TrainDataset(Dataset):
    def __init__(self, data_root='dataset/', data_type='coarse', phase='train'):
        self.size = 32
        # self.resize_width, self.resize_height = 40, 40
        self.data_root = data_root

        self.transform = transforms.Compose([
            # transforms.Resize((self.resize_width, self.resize_height)),
            transforms.RandomCrop(self.size, padding=4),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
        ])
        assert phase in ['train', 'eval']
        self.phase = phase
        self.img_data = np.load(os.path.join(data_root, 'train.npy'))
        if data_type == 'coarse':
            num_ins = 2500
            train_num = 2250
            self.anno = pd.read_csv(os.path.join(data_root, 'train1.csv'))
            label_all = np.array(self.anno['coarse_label'])
            img_id_all = np.array(self.anno['image_id'])
            index = np.argsort(label_all)
            label_sorted = label_all[index]
            img_id_sorted = img_id_all[index]
            self.img_id = []
            self.label_id = []
            for i in range(len(label_all) // num_ins):
                if self.phase == 'train':
                    self.img_id.append(img_id_sorted[num_ins * i:num_ins * i + train_num])
                    self.label_id.append(label_sorted[num_ins * i:num_ins * i + train_num])
                else:
                    self.img_id.append(img_id_sorted[num_ins * i + train_num:num_ins * (i + 1)])
                    self.label_id.append(label_sorted[num_ins * i + train_num:num_ins * (i + 1)])
            self.img_id = np.concatenate(self.img_id, axis=0)
            self.label_id = np.concatenate(self.label_id, axis=0)
        else:
            assert data_type == 'fine'
            num_ins = 500
            train_num = 450
            self.anno = pd.read_csv(os.path.join(data_root, 'train2.csv'))
            label_all = np.array(self.anno['fine_label'])
            img_id_all = np.array(self.anno['image_id'])
            index = np.argsort(label_all)
            label_sorted = label_all[index]
            img_id_sorted = img_id_all[index]
            self.img_id = []
            self.label_id = []
            for i in range(len(label_all) // num_ins):
                if self.phase == 'train':
                    self.img_id.append(img_id_sorted[num_ins * i:num_ins * i + train_num])
                    self.label_id.append(label_sorted[num_ins * i:num_ins * i + train_num])
                else:
                    self.img_id.append(img_id_sorted[num_ins * i + train_num:num_ins * (i + 1)])
                    self.label_id.append(label_sorted[num_ins * i + train_num:num_ins * (i + 1)])
            self.img_id = np.concatenate(self.img_id, axis=0)
            self.label_id = np.concatenate(self.label_id, axis=0)

    def __len__(self):
        return len(self.img_id)

    def __getitem__(self, index):
        img = self.img_data[self.img_id[index]].reshape(3, 32, 32).transpose((1, 2, 0))
        img = Image.fromarray(img)
        label = self.label_id[index]
        img = self.transform(img)
        return img, label

class TestDataset(Dataset):
    def __init__(self, data_root='dataset/'):
        self.size = 32
        self.data_root = data_root
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
        ])
        self.img_data = np.load(os.path.join(data_root, 'test.npy'))


    def __len__(self):
        return len(self.img_data)

    def __getitem__(self, index):
        img = self.img_data[index].reshape(3, 32, 32).transpose((1, 2, 0))
        img = Image.fromarray(img)
        img = self.transform(img)
        return img


class TrainPairDataset(Dataset):
    def __init__(self, data_root='dataset/', data_type='coarse'):
        self.size = 32
        # self.resize_width, self.resize_height = 40, 40
        self.data_root = data_root

        self.transform = transforms.Compose([
            # transforms.Resize((self.resize_width, self.resize_height)),
            transforms.RandomCrop(self.size, padding=4),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
        ])
        self.img_data = np.load(os.path.join(data_root, 'train.npy'))
        if data_type == 'coarse':
            num_ins = 2500
            self.train_num = 2250
            self.anno = pd.read_csv(os.path.join(data_root, 'train1.csv'))
            label_all = np.array(self.anno['coarse_label'])
            img_id_all = np.array(self.anno['image_id'])
            index = np.argsort(label_all)
            label_sorted = label_all[index]
            img_id_sorted = img_id_all[index]
            self.img_id = []
            self.label_id = []
            for i in range(len(label_all) // num_ins):
                self.img_id.append(img_id_sorted[num_ins * i:num_ins * i + self.train_num])
                self.label_id.append(label_sorted[num_ins * i:num_ins * i + self.train_num])
            self.img_id = np.concatenate(self.img_id, axis=0)
            self.label_id = np.concatenate(self.label_id, axis=0)
        else:
            assert data_type == 'fine'
            num_ins = 500
            self.train_num = 450
            self.anno = pd.read_csv(os.path.join(data_root, 'train2.csv'))
            label_all = np.array(self.anno['fine_label'])
            img_id_all = np.array(self.anno['image_id'])
            index = np.argsort(label_all)
            label_sorted = label_all[index]
            img_id_sorted = img_id_all[index]
            self.img_id = []
            self.label_id = []
            for i in range(len(label_all) // num_ins):
                self.img_id.append(img_id_sorted[num_ins * i:num_ins * i + self.train_num])
                self.label_id.append(label_sorted[num_ins * i:num_ins * i + self.train_num])
            self.img_id = np.concatenate(self.img_id, axis=0)
            self.label_id = np.concatenate(self.label_id, axis=0)

    def __len__(self):
        return len(self.img_id)

    def __getitem__(self, index):
        img = self.img_data[self.img_id[index]].reshape(3, 32, 32).transpose((1, 2, 0))
        img = Image.fromarray(img)
        class_num = index // self.train_num
        positive_list = np.arange(self.train_num * class_num, self.train_num * (class_num + 1))
        positive_list = positive_list[positive_list != index]
        positive_index = np.random.choice(positive_list)
        # positive_index = self.img_id[positive_index]
        print(positive_index, index, self.img_id[index], self.img_id[positive_index], self.label_id[index], self.label_id[positive_index])
        img_positive = self.img_data[self.img_id[positive_index]].reshape(3, 32, 32).transpose((1, 2, 0))
        img_positive = Image.fromarray(img_positive)
        label = self.label_id[index]
        img = self.transform(img)
        img_positive = self.transform(img_positive)
        return img, img_positive, label


if __name__ == "__main__":
    dataset = TrainPairDataset(data_type='coarse')
    print(len(dataset))
    d = TrainDataset(data_type='coarse', phase='eval')
    print(d.img_id, 29621 in d.img_id)
    dataloader = DataLoader(dataset=dataset, batch_size=128, num_workers=0, shuffle=True)
    for img, _, label in dataloader:
        print(img.shape, label)

# vim: ts=4 sw=4 sts=4 expandtab
