# test部分
import torch
import Dataloader_new
import My_model_linear_gpu as ff
# import My_model2_linear as ff
from numpy import *
import numpy as np
import Hamming_dis as ham
import time
import scipy.io as sio
import os
import warnings
# from sklearn.preprocessing import OneHotEncoder
warnings.filterwarnings("ignore")
os.environ["CUDA_VISIBLE_DEVICES"] = '1'
use_cuda = torch.cuda.is_available()

num = ff.num
need_train = ff.need_train
batch_size = ff.batch_size

#训练模型
if need_train:
    train_start=time.time()
    epoch_,loss_last = ff.train_data()
    loss_path = "../res_mat_cpu/epoch_loss"+str(num)+".mat"
    sio.savemat(loss_path,{"epoch":np.array(epoch_),"loss":np.array(loss_last)})
    train_end=time.time()
    print("model train time:",train_end-train_start)

fsdh = ff.load_model()
if use_cuda:
    fsdh.cuda()
# 测试集
def test_data():
    print("test data")
    # for name, parameters in fsdh.named_parameters():
    #     print(name, ':', parameters.size(), parameters[:3])
    B_temp_test = torch.Tensor()
    label_test = np.array([])
    YY = torch.Tensor()
    test_dataloader = Dataloader_new.get_test_dataloader()
    for ii, (data, label) in enumerate(test_dataloader):
        data = data.float()
        label = label.float()
#        Y = torch.zeros(size(label, 0), 10)  # 生成一个标签的one-hot编码
#        for i in range(size(label,0)):
#            Y[i][int(label[i])] = 1
#         ohe = OneHotEncoder()
#         ohe.fit([[0],[1],[2],[3],[4],[5],[6],[7],[8],[9]])
#         Y = ohe.transform(label.reshape(-1,1)).toarray()
#         Y = torch.tensor(Y,dtype=torch.float32)
        if use_cuda:
            data, label = data.cuda(), label.cuda()
        B_batch_last = fsdh(data, label, train=False)  # 生成的预测值
        if ii == 1:
            print(B_batch_last[0:5])
        B_batch_last = B_batch_last.cpu()
        B_temp_test = torch.cat([B_temp_test, B_batch_last])  # 1000*32
        # label_test = np.concatenate((label_test,np.array(label)),axis = 0 )
        label = label.cpu()
        YY = torch.cat((YY,label))  # 1000*10
        sim = label.mm(label.t())
        sim = (sim > 0).float()
        sim = (sim - 0.5) * 2
        loss = ff.criterion_fsdh(B_batch_last, sim) / size(data, 0)
        if ii % 100 == 0:
            print(ii, loss.item())
    return B_temp_test,label_test, YY

# 全部数据集
def all_data():
    print("all data")
    # for name, parameters in fsdh.named_parameters():
    #     print(name, ':', parameters.size(), parameters[:3])
    B_temp_exp = torch.Tensor()
    label_exp = []
    YY = torch.Tensor()
    exp_dataloader = Dataloader_new.get_exp_dataloader()
    for ii, (data, label) in enumerate(exp_dataloader):
        data = data.float()
        label = label.float()
#        Y = torch.zeros(size(label, 0), 10)  # 生成一个标签的one-hot编码
#        for i in range(size(label,0)):
#            Y[i][int(label[i])] = 1
#         ohe = OneHotEncoder()
#         ohe.fit([[0],[1],[2],[3],[4],[5],[6],[7],[8],[9]])
#         Y = ohe.transform(label.reshape(-1,1)).toarray()
#         Y = torch.tensor(Y,dtype=torch.float32)1
        if use_cuda:
            data, label = data.cuda(), label.cuda()
        B_batch_last = fsdh(data, label, train=True)  # 生成的预测值
        B_batch_last = B_batch_last.cpu()
        B_temp_exp = torch.cat([B_temp_exp, B_batch_last])
        # label_exp = np.concatenate((label_exp,np.array(label)),axis = 0 )
        label = label.cpu()
        YY = torch.cat((YY, label))
        sim = label.mm(label.t())
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
    # ham_path = "D:\\pycharm\\FSDH\\ham_dis_test2.mat"
    ham_path = "../res_mat_cpu/ham_dis_"+str(num)+".mat"
    sio.savemat(ham_path, {"hamdis": hamdis})

    # label_test_path = "D:\\pycharm\\FSDH\\label_test2.mat"
    # label_exp_path = "D:\\pycharm\\FSDH\\label_exp2.mat"
    label_test_path = "../res_mat_cpu/label_test"+str(num)+".mat"
    label_exp_path = "../res_mat_cpu/label_exp"+str(num)+".mat"
    sio.savemat(label_test_path, {"label_test": np.array(YY_test)})
    sio.savemat(label_exp_path, {"label_exp": np.array(YY_exp)})

    # mAP = cal_new.cal_mAP(YY_exp, YY_test, hamdis)
    # Pre, Re = cal_new.cal_precision_recall(YY_exp, YY_test, hamdis)
    # return mAP, Pre, Re

test_start = time.time()
# mAP, Pre, Re = train()
train()
test_end = time.time()
# plt.figure()
# plt.xlabel("recall")
# plt.ylabel("pre")
# plt.xlim((0,1))
# plt.ylim((0,1))
# plt.title("P-R curve")
# plt.plot(Re, Pre)
# plt.show()
# evalation_path = "eval.mat"
# print("mAP is ",mAP)
# print("Precision")
# print(Pre)
# print("Recall")
# print(Re)
# sio.savemat(evalation_path,{"Precision":Pre,"Recall":Re})
print("final is in ", test_end-test_start)
