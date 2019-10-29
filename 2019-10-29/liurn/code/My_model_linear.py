# Y_matrix是标签
# B = torch.Tensor(UX)
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import torch.nn.functional as Function
import Dataloader_new
from numpy import *
import time

# device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

model_saved_path = "D:\\pycharm\\FSDH\\my_model_layers10_nbits32_lr0.03_epoch100_20191029-1.pt"
# model_saved_path = "fsdh_model_layers10_nbits32_20191027-1.pt"
fsdh_input_dim = 512
fsdh_hidden_1 = 1000
fsdh_hidden_2 = 500
fsdh_out_dim = 32
layers = 10
batch_size = 100
learning_rate = 0.03

class Selfloss(nn.Module):
    def __init__(self, ):
        super(Selfloss, self).__init__()

    def forward(self, B, Sim):
        nbits = size(B, 1)
        loss = torch.frobenius_norm(Sim - 1 / nbits * B.mm(B.t()))
        return loss

class My_model(nn.Module):
    def __init__(self, fsdh_input_dim, fsdh_hidden_1, fsdh_hidden_2, fsdh_out_dim, layers, beta=0.3, gamma=1e-3,
                 alpha=1.0, mu=0.1, nbits=32, batch_size=100):
        super(My_model, self).__init__()
        # FX_matrix是原数据   Y_matrix是标签
        self.layers = layers
        self.beta = beta
        self.gamma = gamma
        self.alpha = alpha
        self.mu = mu
        self.nbits = nbits
        self.batch_size = batch_size
        self.layer1 = nn.Sequential(nn.Linear(fsdh_input_dim, fsdh_hidden_1), nn.BatchNorm1d(fsdh_hidden_1), nn.ReLU())
        self.layer2 = nn.Sequential(nn.Linear(fsdh_hidden_1, fsdh_hidden_2), nn.BatchNorm1d(fsdh_hidden_2), nn.ReLU())
        self.layer3 = nn.Linear(fsdh_hidden_2, fsdh_out_dim)
        # self.W = nn.ParameterList()
        # self.b = nn.ParameterList()
        # for k in range(self.layers):
        self.W = nn.Parameter(torch.eye(self.nbits,fsdh_out_dim,dtype=torch.float32))
        self.b = nn.Parameter(torch.eye(self.nbits, self.batch_size, dtype=torch.float32))

    def forward(self, X, Y, train):
        X = self.layer1(X)
        X = self.layer2(X)
        X = self.layer3(X)
        B = self.W.mm(X.t()) + self.b
        B = Function.tanh(B)
        B = B.t()
        if train:
            for i in range(self.layers):
                P = self.alpha * torch.inverse(X.t().mm(X) + self.gamma * torch.eye(size(X, 1))).mm(X.t()).mm(B.float())
                W = self.beta * torch.inverse(self.beta * (Y.t().mm(Y)).float() +
                                              self.gamma * torch.eye(size(Y, 1)).float()).mm(Y.t().float()).mm(B.float())
                B = Function.tanh(self.alpha * X.mm(P) + self.beta * Y.mm(W))
        else:
            for i in range(self.layers):
                P = self.alpha * torch.inverse(X.t().mm(X) + self.gamma * torch.eye(size(X, 1))).mm(X.t()).mm(B.float())
                B = Function.tanh(self.alpha * X.mm(P))
        return B


mymodel = My_model(fsdh_input_dim, fsdh_hidden_1, fsdh_hidden_2, fsdh_out_dim, layers=layers)
criterion_fsdh = Selfloss()
optimizer = torch.optim.SGD(mymodel.parameters(), lr=learning_rate)

# if torch.cuda.is_available():
#     fsdh.cuda()
#     criterion_fsdh.cuda()

def train_data():
    train_dataloader = Dataloader_new.get_train_dataloader()
    loss_last = []
    loss_sum = 0.0
    epoch_ = []

    for epoch in range(100):
        print('--------epoch', str(epoch))
        for ii, (data, label) in enumerate(train_dataloader):
            Y = torch.zeros(batch_size, 10)  # 生成一个标签的one-hot编码
            # print(data.shape)
            # print(label.shape)
            # if ii == 1:
            #     print(data.shape)
            #     print(label.shape)
            #     print(label)

            for i in range(batch_size):
                Y[i][(label[i])] = 1
            # data_cpu = data.float()
            # if torch.cuda.is_available():
            #     data_gpu = data_cpu.cuda()
            #     Y_gpu = Y.cuda()
            #     binary_gpu = binary.cuda()
            # data = data.float().to(device)
            # binary = binary.to(device)
            # Y = Y.to(device)
            B_batch_last = mymodel(data.float(), Y, train=True)  # 生成的预测值
            sim = Y.mm(Y.t())
            sim = (sim > 0).float()
            sim = (sim - 0.5) * 2
            loss = criterion_fsdh(B_batch_last, sim) / batch_size
            loss_sum = loss_sum + loss
            if ii % 100 == 0:
                print("batch_index: ",ii,"loss:", loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        loss_last.append(float(loss_sum / len(train_dataloader)))
        loss_sum = 0.0
        epoch_.append(epoch)
    plt.figure()
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.title("lr=" + str(learning_rate))
    plt.plot(epoch_, loss_last)
    plt.show()
    torch.save(mymodel, model_saved_path)
    print("Model has been trained")
    params = list(mymodel.named_parameters())
    for name, param in params:
        if name == 'W':
            print(param)
    return epoch_,loss_last

# train_data()