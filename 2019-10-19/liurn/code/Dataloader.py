import math
from scipy.io import loadmat
import torch
from torch.utils import data
from torchvision import transforms as T
from torch.utils.data import DataLoader

import ITQ
root = "E:\北航研究生\实验室\hashing-baselines\datasets\Cifar10-Gist512.mat"

class Cifar_Dataset(data.Dataset):
    def __init__(self, root, B, train):
        m = loadmat(root)
        self.vec = m
        if train == 1:
            self.data = torch.from_numpy(m['X'])[:59000]
            self.label = torch.from_numpy(m['X_class'])[0][:59000]
            self.binary = B[:59000]
        elif train == 2:
            self.data = torch.from_numpy(m['X'])[59000:]
            self.label = torch.from_numpy(m['X_class'])[0][59000:]
            self.binary = B[59000:]
        else:
            self.data = torch.from_numpy(m['X'])[:]
            self.label = torch.from_numpy(m['X_class'])[0][:]
            self.binary = B[:]

    def __getitem__(self, index):
        data = self.data[index]
        label = self.label[index]
        binary = self.binary[index]
        return data, label, binary

    def __len__(self):
        return len(self.data)

B = torch.Tensor(ITQ.get_ITQ_binary())

def get_train_dataloader():
    cifar_dataset_train = Cifar_Dataset(root, B, train=1)
    train_dataloader = DataLoader(cifar_dataset_train, batch_size=100, shuffle=True, num_workers=0)
    return train_dataloader

def get_test_dataloader():
    cifar_dataset_test = Cifar_Dataset(root, B, train=2)
    test_dataloader = DataLoader(cifar_dataset_test, batch_size=1000, shuffle=False, num_workers=0)
    return test_dataloader

def get_exp_dataloader():
    cifar_dataset_exp = Cifar_Dataset(root, B, train=3)
    exp_dataloader = DataLoader(cifar_dataset_exp, batch_size=1000, shuffle=False, num_workers=0)
    return exp_dataloader
# for epoch in range(1):
#     for ii,(data,label,binary) in enumerate(dataloader):
#         print(type(ii))
#         print(label.shape)
#         print(data.shape)
#         print(binary.shape)

