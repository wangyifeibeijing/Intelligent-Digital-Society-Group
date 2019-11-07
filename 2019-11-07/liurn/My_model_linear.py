# Y_matrix是标签
# B = torch.Tensor(UX)
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import torch.nn.functional as Function
import Dataloader_new
import Dataloader
from numpy import *
from sklearn.preprocessing import OneHotEncoder
import warnings
warnings.filterwarnings('ignore')
import argparse
import time

num = 12
checkpoint_num = 1
start_epoch = 0
epoch_num = 200
fsdh_input_dim = 512
fsdh_hidden_1 = 1000
fsdh_hidden_2 = 500
# fsdh_out_dim = np.array([8,16,32,64])
fsdh_out_dim = 32
layers = 10
batch_size = 100
learning_rate = 0.05
need_train = False
resume_train_tag = True
model_saved_path = "G:\\data\\mat_data\\my_model_layers"+str(layers)+"_nbits"+str(fsdh_out_dim)+"_lr"+str(learning_rate)\
                  +"_epoch400"+".pt"  #+str(epoch_num)
# model_saved_path = "G:\\data\\mat_data\\my_model_layers10_nbits32_lr0.005_epoch400_checkpoint1.pt"
model_chp_saved_path = "G:\\data\\mat_data\\my_model_layers" + str(layers) + "_nbits" + str(fsdh_out_dim) + "_lr" + str(
        learning_rate) + "_epoch" + str(epoch_num) + "_checkpoint" + str(checkpoint_num) + ".pt"
print(model_saved_path)
class Selfloss(nn.Module):
    def __init__(self, ):
        super(Selfloss, self).__init__()

    def forward(self, B, Sim):
        nbits = size(B, 1)
        loss = torch.frobenius_norm(Sim - 1 / nbits * B.mm(B.t()))
        return loss

class My_model(nn.Module):
    def __init__(self, fsdh_input_dim, fsdh_hidden_1, fsdh_hidden_2, fsdh_out_dim, layers, beta=0.3, gamma=1e-3,
                 alpha=1.0, mu=0.1, batch_size=100):
        super(My_model, self).__init__()
        self.layers = layers
        self.beta = beta
        self.gamma = gamma
        self.alpha = alpha
        self.mu = mu
        self.batch_size = batch_size
        self.layer1 = nn.Sequential(nn.Linear(fsdh_input_dim, fsdh_hidden_1), nn.BatchNorm1d(fsdh_hidden_1), nn.ReLU())
        self.layer2 = nn.Sequential(nn.Linear(fsdh_hidden_1, fsdh_hidden_2), nn.BatchNorm1d(fsdh_hidden_2), nn.ReLU())
        self.layer3 = nn.Linear(fsdh_hidden_2, fsdh_out_dim)
        self.layer_init = nn.Sequential(nn.Linear(fsdh_input_dim,fsdh_out_dim),nn.Tanh())  # nn.BatchNorm1d(fsdh_out_dim),

    def forward(self, X, Y, train):
        # print(X[:5])
        B = self.layer_init(X)

        X = self.layer1(X)
        X = self.layer2(X)
        X = self.layer3(X)
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
optimizer = torch.optim.SGD(mymodel.parameters(), lr=learning_rate)
criterion_fsdh = Selfloss()

def train_data(model):
    model.train()
    train_dataloader = Dataloader_new.get_train_dataloader()
    loss_last = []
    epoch_ = []
    loss_sum = 0.0
    for epoch in range(start_epoch,start_epoch+epoch_num):
        print('--------epoch', str(epoch))
        for ii, (data, label) in enumerate(train_dataloader):
            # Y = torch.zeros(batch_size, 10)  # 生成一个标签的one-hot编码
            # for i in range(batch_size):
            #     Y[i][int(label[i])] = 1
            ohe = OneHotEncoder()
            ohe.fit([[0],[1],[2],[3],[4],[5],[6],[7],[8],[9]])
            Y = ohe.transform(label.reshape(-1,1)).toarray()
            Y = torch.tensor(Y,dtype=torch.float32)
            B_batch_last = model(data.float(), Y, train=True)  # 生成的预测值
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
    state = {'model':model.state_dict(), 'optimizer':optimizer.state_dict(), 'epoch':epoch, 'loss':loss_last}
    if need_train:
        torch.save(state, model_saved_path)
    elif resume_train_tag:
        torch.save(state, model_chp_saved_path)
    print("Model has been trained")
    return model,epoch_,loss_last

# train_data()

def resume_train():
    checkpoint = torch.load(model_saved_path)
    model = My_model(fsdh_input_dim, fsdh_hidden_1, fsdh_hidden_2, fsdh_out_dim, layers=layers)
    model.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    model.train()
    model,epoch_,loss_last = train_data(model)
    return model,epoch_,loss_last

def load_model():
    checkpoint = torch.load(model_chp_saved_path)
    model = My_model(fsdh_input_dim, fsdh_hidden_1, fsdh_hidden_2, fsdh_out_dim, layers=layers)
    model.load_state_dict(checkpoint['model'])
    model.eval()
    return model

