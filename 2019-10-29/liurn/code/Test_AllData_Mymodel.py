# test部分
import torch
import Dataloader_new
import My_model_linear as ff
from numpy import *
import numpy as np
import cal_new
import Hamming_dis as ham
import time
import matplotlib.pyplot as plt
import scipy.io as sio

train_start=time.clock()
epoch_,loss_last = ff.train_data()
loss_path = "epoch_loss.mat"
sio.savemat(loss_path,{"epoch":np.array(epoch_),"loss":np.array(loss_last)})
train_end=time.clock()
print("model train time:",train_end-train_start)
batch_size = ff.batch_size

model_path = "D:\\pycharm\\FSDH\\my_model_layers10_nbits32_lr0.03_epoch100_20191029-1.pt"
# model_path = "fsdh_model_layers10_nbits32_20191027-1.pt"
fsdh = torch.load(model_path)
fsdh.eval()
# 测试集
def test_data():
    print("test data")
    B_temp_test = torch.Tensor()
    label_test = np.array([])
    YY = torch.Tensor()
    test_dataloader = Dataloader_new.get_test_dataloader()
    for ii, (data, label) in enumerate(test_dataloader):
        Y = torch.zeros(size(label, 0), 10)  # 生成一个标签的one-hot编码
        # print(data.shape)
        # print(label.shape)
        # print(label)
        # if ii == 1:
        #     print(data[0:5])
        #     print(label.shape)
        #     print(label)
        for i in range(size(label,0)):
            Y[i][int(label[i])] = 1
        # Y.int()
        B_batch_last = fsdh(data.float(), Y, train=False)  # 生成的预测值
        if ii == 1:
            print(B_batch_last[0:5])
        B_temp_test = torch.cat([B_temp_test, B_batch_last])  # 1000*32
        label_test = np.concatenate((label_test,np.array(label)),axis = 0 )
        YY = torch.cat((YY,Y))  # 1000*10
        sim = Y.mm(Y.t())
        sim = (sim > 0).float()
        sim = (sim - 0.5) * 2
        loss = ff.criterion_fsdh(B_batch_last, sim) / size(data, 0)
        if ii % 100 == 0:
            print(ii, loss.item())
    return B_temp_test,label_test, YY

# 全部数据集
def all_data():
    print("all data")
    B_temp_exp = torch.Tensor()
    label_exp = []
    YY = torch.Tensor()
    exp_dataloader = Dataloader_new.get_exp_dataloader()
    for ii, (data, label) in enumerate(exp_dataloader):
        Y = torch.zeros(size(label, 0), 10)  # 生成一个标签的one-hot编码
        # print(data.shape)
        # print(label.shape)
        # print(label)
        # if ii == 1:
        #     print(data.shape)
        #     print(label.shape)
        #     print(label)
        for i in range(size(label,0)):
            Y[i][int(label[i])] = 1

        # Y.int()
        # print(Y[:20])
        B_batch_last = fsdh(data.float(), Y, train=True)  # 生成的预测值
        #     print(B_batch_last[0])
        B_temp_exp = torch.cat([B_temp_exp, B_batch_last])
        label_exp = np.concatenate((label_exp,np.array(label)),axis = 0 )
        YY = torch.cat((YY, Y))
        sim = Y.mm(Y.t())
        sim = (sim > 0).float()
        sim = (sim - 0.5) * 2
        loss = ff.criterion_fsdh(B_batch_last, sim) / size(data, 0)
        if ii%100==0:
            print(ii, loss.item())
    print("All data have been processed")
    return B_temp_exp, label_exp, YY

def train():
    B_temp_test, label_test, YY_test = test_data()  # YY_test 1000*10
    B_temp_exp, label_exp, YY_exp = all_data()  # YY_exp 60000*10
    B_last_test = (+1) * (B_temp_test >= 0).float() + \
                  (0) * (B_temp_test < 0).float()
    B_last_exp = (+1) * (B_temp_exp >= 0).float() + \
                 (0) * (B_temp_exp < 0).float()
    print(B_last_exp[:5])
    print(B_last_test[:5])
    hamdis = ham.get_hamming_mat(B_last_test,B_last_exp)
    hamdis = hamdis.astype("uint8")
    ham_path = "D:\\pycharm\\FSDH\\ham_dis_test1.mat"
    sio.savemat(ham_path, {"hamdis": hamdis})

    label_test_path = "D:\\pycharm\\FSDH\\label_test1.mat"
    label_exp_path = "D:\\pycharm\\FSDH\\label_exp1.mat"
    sio.savemat(label_test_path, {"label_test": np.array(YY_test)})
    sio.savemat(label_exp_path, {"label_exp": np.array(YY_exp)})

    mAP = cal_new.cal_mAP(YY_exp, YY_test, hamdis)
    Pre, Re = cal_new.cal_precision_recall(YY_exp, YY_test, hamdis)
    return mAP, Pre, Re

test_start = time.clock()
mAP, Pre, Re = train()

test_end = time.clock()
plt.figure()
plt.xlabel("recall")
plt.ylabel("pre")
plt.xlim((0,1))
plt.ylim((0,1))
plt.title("P-R curve")
plt.plot(Re, Pre)
plt.show()
evalation_path = "eval.mat"
print("mAP is ",mAP)
print("Precision")
print(Pre)
print("Recall")
print(Re)
sio.savemat(evalation_path,{"Precision":Pre,"Recall":Re})
print("final is in ", test_end-test_start)
