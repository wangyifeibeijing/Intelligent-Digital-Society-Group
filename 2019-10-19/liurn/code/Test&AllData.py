# test部分
import torch
import Dataloader
import FSDH_model as ff
from numpy import *
import numpy as np
import cal
import Hamming_dis as ham

ff.train_data()
batch_size = ff.batch_size

model_path = "D:\\pycharm\\FSDH\\fsdh_model_layers30_nbits64.pt"
fsdh = torch.load(model_path)
fsdh.eval()
# 测试集
def test_data():
    print("test data")
    B_temp_test = torch.Tensor()
    label_test = np.array([])
    YY = torch.Tensor()
    test_dataloader = Dataloader.get_test_dataloader()
    for ii, (data, label, binary) in enumerate(test_dataloader):
        Y = torch.zeros(size(label, 0), 10)  # 生成一个标签的one-hot编码
        for i in range(size(label,0)):
            Y[i][int(label[i])] = 1
        B_batch_last = fsdh(data.float(), binary, Y, train=False)  # 生成的预测值
        #     print(B_batch_last[0])
        B_temp_test = torch.cat([B_temp_test, B_batch_last])
        label_test = np.concatenate((label_test,np.array(label)),axis = 0 )
        YY = torch.cat((YY,Y))
        sim = Y.mm(Y.t())
        sim = (sim > 0).float()
        sim = (sim - 0.5) * 2
        loss = ff.criterion_fsdh(B_batch_last, sim) / size(data, 0)
        print(ii, loss.item())
    return B_temp_test,label_test, YY

# 全部数据集
def all_data():
    print("all data")
    B_temp_exp = torch.Tensor()
    label_exp = []
    YY = torch.Tensor()
    exp_dataloader = Dataloader.get_exp_dataloader()
    for ii, (data, label, binary) in enumerate(exp_dataloader):
        Y = torch.zeros(size(label, 0), 10)  # 生成一个标签的one-hot编码
        for i in range(size(label,0)):
            Y[i][int(label[i])] = 1
        # print(Y[:20])
        B_batch_last = fsdh(data.float(), binary, Y, train=True)  # 生成的预测值
        #     print(B_batch_last[0])
        B_temp_exp = torch.cat([B_temp_exp, B_batch_last])
        label_exp = np.concatenate((label_exp,np.array(label)),axis = 0 )
        YY = torch.cat((YY, Y))
        sim = Y.mm(Y.t())
        sim = (sim > 0).float()
        sim = (sim - 0.5) * 2
        loss = ff.criterion_fsdh(B_batch_last, sim) / size(data, 0)
        if ii%10==0:
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
    # print(B_last_test.shape)
    # print(label_test)
    # print("YY_test")
    # print(YY_test)
    # print(B_last_exp.shape)
    # print(label_exp)
    # print("YY_exp")
    # print(YY_exp)
    hamdis = ham.get_hamming_mat(B_last_test[:100],B_last_exp)
    # print(len(label_test))
    # print(label_test.shape)
    # label_test_last = torch.zeros(len(label_test[:10]), 10)  # 生成一个标签的one-hot编码
    # label_test_last = torch.zeros(10, 10)  # 生成一个标签的one-hot编码
    # for i in range(10):
    #     label_test_last[i][label_test[i]] = 1
    # # label_exp_last = torch.zeros(len(label_exp), 10)  # 生成一个标签的one-hot编码
    # label_exp_last = torch.zeros(100, 10)  # 生成一个标签的one-hot编码
    # for i in range(100):
    #     label_exp_last[i][label_exp[i]] = 1
    return cal.cal_mAP(YY_exp,YY_test[:100],hamdis)

ap = train()
print("mAP is ",ap)
