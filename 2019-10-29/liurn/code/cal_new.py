import numpy as np
from numpy import *
import torch

# input
# hamdis是距离矩阵 还没有排序
# train_label test_label是array
def cal_mAP(train_label,test_label,hamdis):
    numtest, numtrain = size(hamdis,0),size(hamdis,1)
    apall = torch.zeros(1,numtest)
    for i in range(numtest):
        if i%100 == 0:
            print(i," ap has been processed")
        y = torch.Tensor(hamdis[i]).int()  # y是一个1*60000的矩阵，表示test中的数据点i到数据集中其他点之间的距离
        y.resize_((1,len(y)))
        x = 0
        p = 0
        temp_test_label = test_label[i].reshape(1,len(test_label[i]))
        new_label = ((temp_test_label.mm(train_label.T))>0).int()
        # new_label = (test_label[i].resize_(1,len(test_label[i]))).mm((train_label).T).float()
        topK = numtrain  # 所有的样本数量
        IX = np.lexsort(np.array(y))
        # y new_label
        # print("begin")
        # print(temp_y[0,10:20])
        # print(temp_label[0,10:20])
        # print(compose[:,:10])
        # temp_0 = compose[0,:]  # 距离
        # temp_0 = temp_0.reshape(1,numtrain)
        # print(temp_0[0,10:20])
        # temp_1 = compose[1,:]  # 相似性
        # temp_1 = temp_1.reshape((1,numtrain))
        # print(temp_1[0,10:20])
        # print(compose[:,:30])
        for j in range(topK):
            # if compose[1][j] == 1:
            if new_label[0,IX[j]] == 1:
                x = x+1
                p = p+x/(j+1)  # x/j是第j个位置上的Precision
                # if x in topR:
                #     x_pre.append(x/(j+1))
                #     x_re.append(x/1000)
        # print(x_pre)
        # print(x_re)
        # preall.append(x_pre)
        # reall.append(x_re)
        if p == 0:
            apall[0][i] = 0
        else:
            apall[0][i] = p/x

    # print("total_good_pairs: ",total_good_pairs)
    # mpre = np.mean(preall,axis=0)
    # mre = np.mean(reall,axis=0)
    temp= np.array(apall[0])
    # print(temp)
    map = mean(temp)
    return map

def cal_precision_recall(train_label,test_label,hamdis):
    numtest, numtrain = size(hamdis, 0), size(hamdis, 1)
    topR = [1, 1000, 2000, 3000, 4000, 5000, 10000, 15000, 20000, 25000, 30000, 35000, 40000, 45000, 50000]
    new_label = ((test_label.mm(train_label.T))>0).int()
    preall = []
    reall = []
    for i in range(numtest):
        y = torch.Tensor(hamdis[i]).int()  # y是一个1*60000的矩阵，表示test中的数据点i到数据集中其他点之间的距离
        y.resize_((1, len(y)))
        IX = np.lexsort(np.array(y))
        # y new_label
        temp_y = y.T[IX].T
        temp_label = new_label[i].T[IX].T
        hamdis[i] = temp_y
        new_label[i] = temp_label

    total_good_pairs = sum(sum(np.array(new_label))).item()

    for i in range(len(topR)):
        if i%100 == 0:
            print(i," pre and re has been processed")
        g = topR[i]
        retrieved_good_pairs = sum(np.sum(np.array(new_label[:,:g]),axis=0)).item()
        # print(retrieved_good_pairs)
        temp = new_label[:,:g]
        row, col = temp.size()
        total_pairs = row * col
        reall.append(retrieved_good_pairs/total_good_pairs)
        preall.append(retrieved_good_pairs/total_pairs)
    preall = np.array(preall)
    reall = np.array(reall)
    return preall,reall
