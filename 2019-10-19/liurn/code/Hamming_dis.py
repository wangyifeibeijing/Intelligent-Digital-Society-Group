import numpy as np
from numpy import *
import math
import torch

bit_in_char = ([0,1,1,2,1,2,2,3,1,2,2,3,2,3,3,4,1,2,2,3,2,3,
                3,4,2,3,3,4,3,4,4,5,1,2,2,3,2,3,3,4,2,3,3,4,
                3,4,4,5,2,3,3,4,3,4,4,5,3,4,4,5,4,5,5,6,1,2,
                2,3,2,3,3,4,2,3,3,4,3,4,4,5,2,3,3,4,3,4,4,5,
                3,4,4,5,4,5,5,6,2,3,3,4,3,4,4,5,3,4,4,5,4,5,
                5,6,3,4,4,5,4,5,5,6,4,5,5,6,5,6,6,7,1,2,2,3,
                2,3,3,4,2,3,3,4,3,4,4,5,2,3,3,4,3,4,4,5,3,4,
                4,5,4,5,5,6,2,3,3,4,3,4,4,5,3,4,4,5,4,5,5,6,
                3,4,4,5,4,5,5,6,4,5,5,6,5,6,6,7,2,3,3,4,3,4,
                4,5,3,4,4,5,4,5,5,6,3,4,4,5,4,5,5,6,4,5,5,6,
                5,6,6,7,3,4,4,5,4,5,5,6,4,5,5,6,5,6,6,7,4,5,
                5,6,5,6,6,7,5,6,6,7,6,7,7,8])

def hamming_dis(S1,S2):
    return sum([(int(ch1) ^ int(ch2)) for ch1, ch2 in zip(S1, S2)])

# def compactbit(B):
#     nSamples, nbits = B.size()
#     nwords = nbits/8
#     cb = torch.zeros(nSamples,nwords)
#     for j in range(nbits):
#         w = ceil(j/8)
#         temp = B[:,]
#         cb[]
#     return cb

def get_hamming_mat(B_last_test, B_last_exp):
    # print(size(B_last_test,0))
    Dish = np.zeros((size(B_last_test,0),size(B_last_exp,0)))
    print("get hamming dis mat")
    for i in range(size(B_last_test,0)):
        print(i,"-------")
        for j in range(size(B_last_exp,0)):
            if j%10000==0:
                print(j)
            res = hamming_dis(B_last_test[i],B_last_exp[j])
            Dish[i][j] = res
    # print(Dish)
    print("get hamming dis mat successfully")
    return Dish

