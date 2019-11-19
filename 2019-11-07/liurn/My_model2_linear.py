# Y_matrix是标签
# B = torch.Tensor(UX)
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import torch.nn.functional as Function
import Dataloader_new
from numpy import *
import os
# import argparse
import time
# import torchsnooper
# from sklearn.preprocessing import OneHotEncoder
import warnings

warnings.filterwarnings("ignore")
# device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

# model_saved_path = "D:\\pychdarm\\FSDH\\my_model_layers10_nbits32_lr0.005_epoch500_20191030-3.pt"
num = 33
start_epoch = 1
epoch_num = 2
fsdh_input_dim = 500
fsdh_hidden_1 = 1000
fsdh_hidden_2 = 500
fsdh_out_dim = 32
layers = 3
batch_size = Dataloader_new.get_batchsize()
learning_rate = 0.05
checkpoint_num = 0
need_train = True
model_saved_path = "../res_mat_cpu/my_model" + str(num) + "_layers" + str(layers) + "_nbits" + str(
    fsdh_out_dim) + "_lr" + str(learning_rate) + "_epoch" + str(epoch_num) + ".pt"
print(model_saved_path)
os.environ["CUDA_VISIBLE_DEVICES"] = '1'
use_cuda = torch.cuda.is_available()
print("use_cuda: ", use_cuda)


class Selfloss(nn.Module):
    def __init__(self, ):
        super(Selfloss, self).__init__()

    def forward(self, B, Sim):
        nbits = size(B, 1)
        loss = torch.frobenius_norm(Sim - 1 / nbits * B.mm(B.t()))
        return loss


class My_model(nn.Module):
    def __init__(self, fsdh_input_dim, fsdh_hidden_1, fsdh_hidden_2, fsdh_out_dim, batch_size, layers, beta=2.0,
                 gamma=1e-3,
                 alpha=1.0, mu=0.):
        super(My_model, self).__init__()
        # FX_matrix是原数据   Y_matrix是标签
        self.layers = layers
        self.beta = beta
        self.gamma = gamma
        self.alpha = alpha
        self.mu = mu
        self.batch_size = batch_size
        self.layer1 = nn.Sequential(nn.Linear(fsdh_input_dim, fsdh_hidden_1), nn.BatchNorm1d(fsdh_hidden_1), nn.ReLU())
        self.layer2 = nn.Sequential(nn.Linear(fsdh_hidden_1, fsdh_hidden_2), nn.BatchNorm1d(fsdh_hidden_2), nn.ReLU())
        self.layer3 = nn.Linear(fsdh_hidden_2, fsdh_out_dim)
        self.layer_init = nn.Sequential(nn.Linear(fsdh_input_dim, fsdh_out_dim), nn.BatchNorm1d(fsdh_out_dim))
        self.layer_init1 = nn.Tanh()
        # self.diag =  nn.ParameterList([nn.Parameter(torch.eye(batch_size)) for i in range(layers+1)])
        self.vec = nn.ParameterList([nn.Parameter(torch.ones(fsdh_out_dim,1)) for i in range(layers+1)])
    def forward(self, X, Y, train):
        B = self.layer_init(X)
        martix1_temp = torch.diagflat(self.vec[0])
        B = self.layer_init1(B.mm(martix1_temp.mm(martix1_temp.t())))
        X = self.layer1(X)
        X = self.layer2(X)
        X = self.layer3(X)
        if train:
            for i in range(self.layers):
                martix_temp = torch.diagflat(self.vec[i+1])
                martix = martix_temp.mm(martix_temp.t())
                P = self.alpha * torch.inverse(X.t().mm(X) + self.gamma * torch.eye(size(X, 1))).mm(X.t()).mm(
                    B.float())  # .cuda()
                W = self.beta * torch.inverse(
                    self.beta * (Y.t().mm(Y)).float() + self.gamma * torch.eye(size(Y, 1)).float()).mm(
                    Y.t().float()).mm(B.float())
                B_temp = (self.alpha * X.mm(P) + self.beta * Y.mm(W)).mm(martix)
                B = Function.tanh(B_temp)
        else:
            for i in range(self.layers):
                martix_temp = torch.diagflat(self.vec[i + 1])
                martix = martix_temp.mm(martix_temp.t())
                P = self.alpha * torch.inverse(X.t().mm(X) + self.gamma * torch.eye(size(X, 1))).mm(X.t()).mm(B.float())
                B_temp = (self.alpha * X.mm(P)).mm(martix)
                B = Function.tanh(B_temp)
        return B


# parser = argparse.ArgumentParser(description='CIFAR10 Training')
# parser.add_argument('--lr',default=learning_rate,type=float, help='learning rate')
# parser.add_argument('--resume','-r', action='store_true', help='resume from checkpoint')
# args = parser.parse_args()

# if args.resume:
#     print('==> Resuming from checkpoint..')
#     checkpoint = torch.load(model_saved_path)
#     mymodel = My_model(fsdh_input_dim, fsdh_hidden_1, fsdh_hidden_2, fsdh_out_dim, layers=layers)
#     optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)
#     mymodel.load_state_dict(checkpoint['model'])
#     optimizer.load_state_dict(checkpoint['optimizer'])
#     start_epoch = checkpoint['epoch']
#     loss_last = checkpoint['loss']
# else:
#     print('==> Building model..')

mymodel = My_model(fsdh_input_dim, fsdh_hidden_1, fsdh_hidden_2, fsdh_out_dim, batch_size=batch_size, layers=layers)
optimizer = torch.optim.SGD(mymodel.parameters(), lr=learning_rate)
criterion_fsdh = Selfloss()

if use_cuda:
    mymodel.cuda()

# @torchsnooper.snoop()
def train_data():
    mymodel.train()
    train_dataloader = Dataloader_new.get_train_dataloader()
    loss_last = []
    epoch_ = []
    loss_sum = 0.0
    for epoch in range(start_epoch, start_epoch + epoch_num):
        # if epoch<5:
        print('--------epoch', str(epoch))
        epoch_st = time.time()
        for ii, (data, label) in enumerate(train_dataloader):
            data = data.float()
            label = label.float()
            # Y = torch.zeros(batch_size, 10)  # 生成一个标签的one-hot编码
            # for i in range(batch_size):
            #     Y[i][int(label[i])] = 1
            #            ohe = OneHotEncoder()
            #            ohe.fit([[0],[1],[2],[3],[4],[5],[6],[7],[8],[9]])
            #            Y = ohe.transform(label.reshape(-1,1)).toarray()
            #            Y = torch.tensor(Y,dtype=torch.float32)
            if use_cuda:
                data, label = data.cuda(), label.cuda()
            round_st = time.time()
            B_batch_last = mymodel(data, label, train=True)  # 生成的预测值
            round_et = time.time()
            norm2 = torch.norm(label,p=2,dim=1,keepdim=True)
            label1 = label/norm2
            sim = label1.mm(label1.t())
            # sim = (sim > 0).float()
            sim = (sim - 0.5) * 2
            loss = criterion_fsdh(B_batch_last, sim) / batch_size
            loss_sum = loss_sum + loss
            if ii % 100 == 0:
                print("batch_index: ", ii, "loss:", loss.item(), "time: ", round_et - round_st)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        loss_last.append(float(loss_sum / len(train_dataloader)))
        loss_sum = 0.0
        epoch_.append(epoch)
        epoch_et = time.time()
        print("epoch time: ", epoch_et - epoch_st)
        global checkpoint_num
        for name, parameters in mymodel.named_parameters():
            print(name, ':', parameters.size(), parameters[:3])
    #        if checkpoint_num>0:
    #            if epoch%1000==0:
    #                path_temp = model_saved_path+'_ckp'+str(checkpoint_num)
    #                checkpoint_num = checkpoint+1
    #                state_temp = {'model':mymodel.state_dict(), 'optimizer':optimizer.state_dict(), 'epoch':epoch, 'loss':loss_last}
    #                torch.save(state_temp, path_temp)

    #    plt.figure()
    #    plt.xlabel("epoch")
    #    plt.ylabel("loss")
    #    plt.title("lr=" + str(learning_rate))
    #    plt.plot(epoch_, loss_last)
    #    plt.show()
    state = {'model': mymodel.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': epoch, 'loss': loss_last}
    torch.save(state, model_saved_path)
    print("Model has been trained")
    return epoch_, loss_last


# train_data()

def resume_train():
    checkpoint = torch.load(model_saved_path)
    model = My_model(fsdh_input_dim, fsdh_hidden_1, fsdh_hidden_2, fsdh_out_dim, layers=layers)
    model.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    model.train()
    train_data()
    return model


def load_model():
    checkpoint = torch.load(model_saved_path)
    model = My_model(fsdh_input_dim, fsdh_hidden_1, fsdh_hidden_2, fsdh_out_dim, layers=layers, batch_size=batch_size)
    #    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    model.load_state_dict(checkpoint['model'])
    #    optimizer.load_state_dict(checkpoint['optimizer'])
    #    epoch = checkpoint['epoch']
    #    loss = checkpoint['loss']
    model.eval()
    return model

