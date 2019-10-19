#ITQ模块
import math
from numpy import *
import numpy as np
import torch
import torch.nn as nn
from scipy.io import loadmat
nbits = 32
m = loadmat("E:\北航研究生\实验室\hashing-baselines\datasets\Cifar10-Gist512.mat")

#  pca函数
def pca(dataMat, topNfeat=nbits):    # topNfeat 降维后的维度
    meanVals = mean(dataMat, axis=0)     # 按列求均值，即每一列求一个均值，不同的列代表不同的特征
    # print meanVals
    meanRemoved = dataMat - meanVals   # remove mean     # 去均值，将样本数据的中心点移到坐标原点
    print(meanRemoved)
    covMat = cov(meanRemoved, rowvar=0)         # 计算协方差矩阵
    # print covMat
    eigVals, eigVects = linalg.eig(mat(covMat))   # 计算协方差矩阵的特征值和特征向量
    # print eigVals
    # print eigVects
    eigValInd = argsort(eigVals)            # sort, sort goes smallest to largest  #排序，将特征值按从小到大排列
    # print eigValInd
    eigValInd = eigValInd[:-(topNfeat+1):-1]  # cut off unwanted dimensions      #选择维度为topNfeat的特征值
    # print eigValInd
    redEigVects = eigVects[:,eigValInd]       # reorganize eig vects largest to smallest   #选择与特征值对应的特征向量
    print(redEigVects)
    lowDDataMat = meanRemoved * redEigVects   # transform data into new dimensions    #将数据映射到新的维度上，lowDDataMat为降维后的数据
    print(lowDDataMat)
    reconMat = (lowDDataMat * redEigVects.T) + meanVals         # 对原始数据重构，用于测试
    print(reconMat)
    return lowDDataMat, reconMat



# 迭代优化ITQ
def get_ITQ_binary():
    r = torch.randn(nbits, nbits)
    R = mat(r)
    # print(R.shape)
    U, Sigma, VT = linalg.svd(R)  # sigma只返回奇异值
    R = U[:, 0:nbits]  # R是初始化的旋转矩阵
    lowDDataMat, reconMat = pca(m['X'])
    for ii in range(50):
        Z = lowDDataMat * R
        UX = (-1)*(Z<0)+\
             (+1)*(Z>=0)
    #     UX = mat(temp1)
        C = UX.transpose() * lowDDataMat
        U1, Sigma, VT1 = linalg.svd(C)
        R = VT1 * U1.transpose()
    #     print('iter:%d, loss:%f\n',iter,sum(np.sum(np.array(UX)-Z)))
    #     l = nn.MSELoss()
    #     print(l(torch.Tensor(UX),torch.Tensor(Z)))
    #     print(UX[:1])
    #     print('iter:%d, loss:%f\n',ii,mean(np.sum(np.square(UX-Z))))
    print("ITQ finished")
    R_last = R  # 旋转矩阵
    return UX

# get_ITQ_binary()
