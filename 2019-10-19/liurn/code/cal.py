from numpy import *
import numpy as np
import torch

# input
# hamdis是距离矩阵 还没有排序
# train_label test_label是array
def cal_mAP(train_label,test_label,hamdis):
    numtest, numtrain = size(hamdis,0),size(hamdis,1)
    apall = torch.zeros(1,numtest)
    for i in range(numtest):
        y = torch.Tensor(hamdis[i]) # y是一个1*60000的矩阵，表示test中的数据点i到数据集中其他点之间的距离
        y.resize_((1,len(y)))
        x = 0
        p = 0
        new_label = np.array(test_label[i]).dot(np.array(train_label).T)
        new_label = torch.tensor(new_label)
        new_label.resize_((1,len(new_label)))
        print(new_label.shape)
        topK = numtrain  # 所有的样本数量
        compose = torch.cat((y,new_label))
        compose = np.array(compose)
        compose = compose.T[np.lexsort(compose[::-1,:])].T
        print(compose[:,:20])
        for j in range(topK):
            if compose[1][j] == 1:
                x = x+1
                p = p+x/(j+1)  # x/j是第j个位置上的Precision
        if p == 0:
            apall[0][i] = 0
        else:
            apall[0][i] = p/x
    temp= np.array(apall[0])
    print(temp)
    ap = mean(temp)
    return ap
