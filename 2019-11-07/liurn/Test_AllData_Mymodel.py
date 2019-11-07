# test部分
import torch
import Dataloader_new
import Dataloader
import My_model_linear as ff
from numpy import *
import numpy as np
import cal_new
import Hamming_dis as ham
import time
import matplotlib.pyplot as plt
import scipy.io as sio
from sklearn.preprocessing import OneHotEncoder
import warnings
warnings.filterwarnings('ignore')
num = ff.num
need_train = ff.need_train
resume_train_tag = ff.resume_train_tag
checkpoint_num = ff.checkpoint_num
batch_size = ff.batch_size
# 训练模型
if need_train:
    print('need_train')
    model_temp = ff.mymodel
    train_start=time.clock()
    mymodel, epoch_, loss_last = ff.train_data(model_temp)
    # mymodel.eval()
    loss_path = "G:\\data\\mat_data\\epoch_loss"+str(num)+".mat"
    # loss_path = "test2.mat"
    sio.savemat(loss_path,{"epoch":np.array(epoch_),"loss":np.array(loss_last)})
    train_end=time.clock()
    print("model train time:",train_end-train_start)
elif resume_train_tag:
    print('resume_train',resume_train_tag)
    train_start = time.clock()
    mymodel, epoch_, loss_last = ff.resume_train()
    # mymodel.eval()
    loss_path = "G:\\data\\mat_data\\epoch_loss" + str(num) + "_checkpoint" + str(checkpoint_num) + ".mat"
    # loss_path = "test2.mat"
    sio.savemat(loss_path, {"epoch": np.array(epoch_), "loss": np.array(loss_last)})
    train_end = time.clock()
    print("model train time:", train_end - train_start)

fsdh = ff.load_model()

# 测试集
def test_data():
    print("test data")
    B_temp_test = torch.Tensor()
    label_test = np.array([])
    YY_test = torch.Tensor()
    test_dataloader = Dataloader_new.get_test_dataloader()
    for ii, (data, label) in enumerate(test_dataloader):
        # Y = torch.zeros(batch_size, 10)  # 生成一个标签的one-hot编码
        # for i in range(size(label,0)):
        #     Y[i][int(label[i])] = 1
        ohe = OneHotEncoder()
        ohe.fit([[0], [1], [2], [3], [4], [5], [6], [7], [8], [9]])
        Y = ohe.transform(label.reshape(-1, 1)).toarray()
        Y = torch.tensor(Y, dtype=torch.float32)
        B_batch_last = fsdh(data.float(), Y, train=False)  # 生成的预测值
        if ii == 1:
            print(Y[:5])
        B_temp_test = torch.cat([B_temp_test, B_batch_last])  # 1000*32
        label_test = np.concatenate((label_test,np.array(label)),axis = 0 )
        YY_test = torch.cat((YY_test,Y))  # 1000*10
        sim = Y.mm(Y.t())
        sim = (sim > 0).float()
        sim = (sim - 0.5) * 2
        loss = ff.criterion_fsdh(B_batch_last, sim) / batch_size
        if ii % 100 == 0:
            print(ii, loss.item())
    return B_temp_test,label_test, YY_test

# 全部数据集
def all_data():
    print("all data")
    B_temp_exp = torch.Tensor()
    label_exp = []
    YY_exp = torch.Tensor()
    exp_dataloader = Dataloader_new.get_exp_dataloader()
    for ii, (data, label) in enumerate(exp_dataloader):
        # Y = torch.zeros(batch_size, 10)  # 生成一个标签的one-hot编码
        # for i in range(size(label,0)):
        #     Y[i][int(label[i])] = 1
        ohe = OneHotEncoder()
        ohe.fit([[0], [1], [2], [3], [4], [5], [6], [7], [8], [9]])
        Y = ohe.transform(label.reshape(-1, 1)).toarray()
        Y = torch.tensor(Y, dtype=torch.float32)
        B_batch_last = fsdh(data.float(), Y, train=True)  # 生成的预测值

        B_temp_exp = torch.cat([B_temp_exp, B_batch_last])
        label_exp = np.concatenate((label_exp,np.array(label)),axis = 0 )
        YY_exp = torch.cat((YY_exp, Y))
        sim = Y.mm(Y.t())
        sim = (sim > 0).float()
        sim = (sim - 0.5) * 2
        loss = ff.criterion_fsdh(B_batch_last, sim) / batch_size
        if ii%100==0:
            print(ii, loss.item())
    print("All data have been processed")
    return B_temp_exp, label_exp, YY_exp

def train():
    B_temp_test, label_test, YY_test = test_data()  # YY_test 1000*10
    # print(YY_test[:20])
    B_temp_exp, label_exp, YY_exp = all_data()  # YY_exp 60000*10
    B_last_test = (+1) * (B_temp_test >= 0).float() + \
                  (0) * (B_temp_test < 0).float()
    B_last_exp = (+1) * (B_temp_exp >= 0).float() + \
                 (0) * (B_temp_exp < 0).float()
    print(B_last_exp[:5])
    print(B_last_test[:5])
    hamdis = ham.get_hamming_mat(B_last_test,B_last_exp).astype('uint8')
    # hamdis = hamdis.astype("uint8")
    if need_train:
        print("need train")
        ham_path = "G:\\data\\mat_data\\ham_dis_"+str(num)+".mat"
        sio.savemat(ham_path, {"hamdis": hamdis})
        label_test_path = "G:\\data\\mat_data\\test_label_"+str(num)+".mat"
        label_exp_path = "G:\\data\\mat_data\\database_label_"+str(num)+".mat"
        sio.savemat(label_test_path, {"label_test": np.array(YY_test)})
        sio.savemat(label_exp_path, {"label_exp": np.array(YY_exp)})
    elif resume_train_tag:
        print("resume train")
        ham_path = "G:\\data\\mat_data\\ham_dis_" + str(num) + "_checkpoint" + str(checkpoint_num) + ".mat"
        sio.savemat(ham_path, {"hamdis": hamdis})
        label_test_path = "G:\\data\\mat_data\\test_label_" + str(num) + "_checkpoint" + str(checkpoint_num) + ".mat"
        label_exp_path = "G:\\data\\mat_data\\database_label_" + str(num) + "_checkpoint" + str(checkpoint_num) + ".mat"
        sio.savemat(label_test_path, {"label_test": np.array(YY_test)})
        sio.savemat(label_exp_path, {"label_exp": np.array(YY_exp)})
    else:
        print("only use model")
        ham_path = "G:\\data\\mat_data\\ham_dis_" + str(num) + "_checkpoint" + str(checkpoint_num) + ".mat"
        sio.savemat(ham_path, {"hamdis": hamdis})
        label_test_path = "G:\\data\\mat_data\\test_label_" + str(num) + "_checkpoint" + str(checkpoint_num) + ".mat"
        label_exp_path = "G:\\data\\mat_data\\database_label_" + str(num) + "_checkpoint" + str(checkpoint_num) + ".mat"
        sio.savemat(label_test_path, {"label_test": np.array(YY_test)})
        sio.savemat(label_exp_path, {"label_exp": np.array(YY_exp)})
    # mAP = cal_new.cal_mAP(YY_exp, YY_test, hamdis)
    # Pre, Re = cal_new.cal_precision_recall(YY_exp, YY_test, hamdis)
    # return mAP, Pre, Re

test_start = time.clock()
# mAP, Pre, Re = train()
train()
test_end = time.clock()
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
