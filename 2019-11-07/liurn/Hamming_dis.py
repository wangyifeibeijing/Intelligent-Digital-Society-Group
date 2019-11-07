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
    res = 0
    for ch1, ch2 in zip(S1,S2):
        res = res + bit_in_char[int(ch1) ^ int(ch2)]
    return res

def compactbit(B):
    B = np.array(B).astype(int)
    nSamples, nbits = size(B,0),size(B,1)
    # print(nSamples,nbits)
    nwords = int(nbits/8)
    cb = torch.zeros((nSamples,nwords))
    for i in range(nSamples):
        for j in range(nwords):
            # w = ceil(j/8)
            temp = ''.join(str(x) for x in B[i,j*8:j*8+8])
            temp = int(temp,2)
            cb[i,j]= temp
    print(cb)
    return cb

def get_hamming_mat(B_last_test, B_last_exp):
    print(B_last_test.shape)
    print(B_last_exp.shape)
    print(B_last_test[:1])
    print(B_last_exp[:10])
    # B_last_test = compactbit(B_last_test)
    # B_last_exp = compactbit(B_last_exp)
    # print(size(B_last_test,0))
    Dish = np.zeros((size(B_last_test,0),size(B_last_exp,0)),dtype='uint8')  # .astype(int)
    print("get hamming dis mat")
    for i in range(size(B_last_test,0)):
        # print(i,"-------")
        temp1 = np.array(B_last_test[i]).astype(int)
        temp2 = np.array(B_last_exp).astype(int)
        Dish[i] = np.sum((temp1 ^ temp2), axis=1)

    # print(Dish)
    print("get hamming dis mat successfully")
    print(Dish[:1,:10])
    return Dish

def test():
    B1 = [[1,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0],[1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]]
    B2 = [[0,1,0,0,0,0,0,0,1,0,0,0,0,0,0,0],[0,0,1,1,0,0,0,0,1,0,0,0,0,0,0,0]]
    dish = get_hamming_mat(B1,B2)
    return dish

# dish = test()
# print(dish)