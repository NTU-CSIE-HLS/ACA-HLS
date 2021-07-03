from bfv_rns import *
from data import *
import pickle
import datetime
import pickle
import gc
import copy
import defs
import sys
import numpy as np
from multiprocessing import Pool
import socket
from util import *

with open("test.txt", "rb") as fp:   # Unpickling
    context = pickle.load(fp)

with open("encrypt8192.txt", "rb") as fp:   # Unpickling
    ct_ = pickle.load(fp)

# print((context.encryptor[0].rlk[0][0][0].poly[0]))
# print((ct_[0][0][0][0].poly[0]))
# print((context.encryptor[0].galois[0][0][0][0].poly[0]))

# static const int rlk[4][7][2][7][8192]
rlk = []
for i in range(4):
    for j in range(7):
        for k in range(2):
            for l in range(7):
                for x in range(8192):
                    rlk.append(context.encryptor[i].rlk[j][k][l].poly[x])


# static const int ct[25][4][2][7][8192]
ct = []
for i in range(25):
    for j in range(4):
        for k in range(2):
            for l in range(7):            
                for x in range(8192):
                    ct.append(ct_[i][j][k][l].poly[x])

# static const int gk[4][25][7][2][7][8192]
gk = []
for i in range(4):
    for j in range(25):
        for k in range(7):
            for l in range(2):
                for x in range(7):
                    for y in range(8192):
                        gk.append(context.encryptor[i].galois[j][k][l][x].poly[y])

# send to server
print('Transforming data to C array ...')

rlk_arr = List2CArray(rlk)
ct_arr  = List2CArray(ct)
gk_arr  = List2CArray(gk)

print('Connecting to server ...')

ip, port = '140.112.31.159', 17777
sckt = startSocket(ip, port)

print('Start sending data to server ...')
sendData(sckt, rlk_arr)
sendData(sckt, ct_arr)
sendData(sckt, gk_arr)

ret_bytes = receiveData(sckt, num_int=4*2*7*8192)
print('Receive result from server!')
print('Transforming result to Python list ...')

# note: ret is a 1D list
ret = CArray2List(ret_bytes, num_int=4*2*7*8192)

# receive from client
# received ct should be: int ret[4][2][7][8192]
# ret = [[[[0]*8192]*7]*2]*4 # TODO


closeSocket(sckt)
exit()


# packed into ct form
ct_ret = []
for i in ret:
    temp1 = []
    for j in i:
        temp2 = []
        for k, p in zip(j, context.encryptor[0].q_i):
            temp2.append(polynomial(k, p))
        temp1.append(temp2)
    ct_ret.append(temp1)

"""
# 10 min decrypt and decode
pt  = context.decrypt(ct_ret)
out = context.decode_and_reconstruct(pt)

result = []
for i in range(10):
    result.append(out[16*i]/(1<<51))
print(result)
print(result.index(max(result)))
"""
