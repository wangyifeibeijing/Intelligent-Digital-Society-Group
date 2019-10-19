# Y_matrix是标签
# B = torch.Tensor(UX)
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import torch.nn.functional as Function
import Dataloader
from numpy import *

model_saved_path = "D:\\pycharm\\FSDH\\fsdh_model_layers30_nbits64.pt"
fsdh_input_dim = 512
fsdh_hidden_1 = 1000
fsdh_hidden_2 = 500
fsdh_out_dim = 64
layers = 30
batch_size = 100
learning_rate = 0.08

class Selfloss(nn.Module):
    def __init__(self, ):
        super(Selfloss, self).__init__()

    def forward(self, B, Sim):
        nbits = size(B, 1)
        loss = torch.frobenius_norm(Sim - 1 / nbits * B.mm(B.t()))
        return loss

class FSDH(nn.Module):
    def __init__(self, fsdh_input_dim, fsdh_hidden_1, fsdh_hidden_2, fsdh_out_dim, layers, beta=0.3, gamma=1e-3,
                 alpha=1.0, mu=0.1):
        super(FSDH, self).__init__()
        # FX_matrix是原数据   Y_matrix是标签
        self.layers = layers
        self.beta = beta
        self.gamma = gamma
        self.alpha = alpha
        self.mu = mu
        self.layer1 = nn.Sequential(nn.Linear(fsdh_input_dim, fsdh_hidden_1), nn.BatchNorm1d(fsdh_hidden_1), nn.ReLU())
        self.layer2 = nn.Sequential(nn.Linear(fsdh_hidden_1, fsdh_hidden_2), nn.BatchNorm1d(fsdh_hidden_2), nn.ReLU())
        self.layer3 = nn.Linear(fsdh_hidden_2, fsdh_out_dim)

    def forward(self, X, B, Y, train):
        X = self.layer1(X)
        X = self.layer2(X)
        X = self.layer3(X)
        if train:
            for i in range(self.layers):
                P = self.alpha * torch.inverse(X.t().mm(X) + self.gamma * torch.eye(size(X, 1))).mm(X.t()).mm(B.float())
                W = self.beta * torch.inverse(self.beta * (Y.t().mm(Y)).float() +
                                              self.gamma * torch.eye(size(Y, 1)).float()).mm(Y.t().float()).mm(B.float())
                #                 temp = self.beta*Y.mm(W) + self.alpha * X.mm(P)
                #                 gradientB = (-1)*(B<-1).float()+\
                #                             (+1)*((B>=-1)&(B<=0)).float()+\
                #                             (-1)*((B>0)&(B<1)).float()+\
                #                             (+1)*(B>=1).float()
                #                 B = 1.0/(self.beta+self.alpha)*(temp - gradientB * self.mu / 2)
                B = Function.tanh(self.alpha * X.mm(P) + self.beta * Y.mm(W))
        else:
            for i in range(self.layers):
                P = self.alpha * torch.inverse(X.t().mm(X) + self.gamma * torch.eye(size(X, 1))).mm(X.t()).mm(B.float())
                B = Function.tanh(self.alpha * X.mm(P))

                # temp = self.alpha * X.mm(P)
                # gradientB = (-1) * (B < -1).float() + \
                #             (+1) * ((B >= -1) & (B <= 0)).float() + \
                #             (-1) * ((B > 0) & (B < 1)).float() + \
                #             (+1) * (B >= 1).float()
                # B = 1.0 / self.alpha * (temp - gradientB * self.mu / 2)
        return B

fsdh = FSDH(fsdh_input_dim, fsdh_hidden_1, fsdh_hidden_2, fsdh_out_dim, layers=layers)
criterion_fsdh = Selfloss()
optimizer = torch.optim.SGD(fsdh.parameters(), lr=learning_rate)

def train_data():
    train_dataloader = Dataloader.get_train_dataloader()
    loss_last = []
    loss_sum = 0.0
    epoch_ = []
    plt.figure()
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.title("lr=" + str(learning_rate))
    for epoch in range(300):
        print('--------epoch', str(epoch))
        for ii, (data, label, binary) in enumerate(train_dataloader):
            Y = torch.zeros(batch_size, 10)  # 生成一个标签的one-hot编码
            for i in range(batch_size):
                Y[i][int(label[i])] = 1
            B_batch_last = fsdh(data.float(), binary, Y, train=True)  # 生成的预测值
            sim = Y.mm(Y.t())
            sim = (sim > 0).float()
            sim = (sim - 0.5) * 2
            loss = criterion_fsdh(B_batch_last, sim) / batch_size
            loss_sum = loss_sum + loss
            if ii % 100 == 0:
                print(ii, loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        loss_last.append(float(loss_sum / len(train_dataloader)))
        loss_sum = 0.0
        epoch_.append(epoch)
    plt.plot(epoch_, loss_last)
    plt.show()
    torch.save(fsdh, model_saved_path)
    print("Model has been trained")

# train_data()