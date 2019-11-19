import math
from scipy.io import loadmat
import torch
import numpy as np
from torch.utils import data
# from torchvision import transforms as T
from torch.utils.data import DataLoader
# import ITQ
batch_size = 500
db_name = 'nuswide'
class Cifar_Dataset(data.Dataset):
    def __init__(self, db_root, train):
        db = loadmat(db_root)
        if train == 1:
            self.data = torch.from_numpy(np.array(db['train_data'],dtype=float))
            self.label = torch.from_numpy(np.array(db['train_label'],dtype=float))
        elif train == 2:
            self.data = torch.from_numpy(np.array(db['test_data'],dtype=float))
            self.label = torch.from_numpy(np.array(db['test_label'],dtype=float))
        else:
            self.data = torch.from_numpy(np.array(db['train_data'],dtype=float))
            self.label = torch.from_numpy(np.array(db['train_label'],dtype=float))

    def __getitem__(self, index):
        data = self.data[index]
        label = self.label[index]
        return data, label

    def __len__(self):
        return len(self.data)

# B = torch.Tensor(ITQ.get_ITQ_binary())
if db_name == 'cifar':
    db_root = "../Datasets/db_data.mat"
elif db_name == 'nuswide':
    # db_root = "../Datasets/db_data_NUS-WIDE.mat"
    db_root = "../db_data_NUS-WIDE2.mat"
def get_train_dataloader():
    cifar_dataset_train = Cifar_Dataset(db_root, train=1)
    train_dataloader = DataLoader(cifar_dataset_train, batch_size=batch_size, shuffle=True, num_workers=0) # ,drop_last=True
    return train_dataloader

def get_test_dataloader():
    cifar_dataset_test = Cifar_Dataset(db_root, train=2)
    test_dataloader = DataLoader(cifar_dataset_test, batch_size=batch_size, shuffle=False, num_workers=0) # ,drop_last=True
    return test_dataloader

def get_exp_dataloader():
    cifar_dataset_exp = Cifar_Dataset(db_root, train=3)
    exp_dataloader = DataLoader(cifar_dataset_exp, batch_size=batch_size, shuffle=False, num_workers=0) # ,drop_last=True
    return exp_dataloader

def get_batchsize():
    return batch_size
# dataloader = get_train_dataloader()
# for epoch in range(1):
#     for ii,(data,label) in enumerate(dataloader):
#         print(ii)
#         print(label.shape)
#         print(data.shape)

