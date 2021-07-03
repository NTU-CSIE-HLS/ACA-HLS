from bfv_rns import *
from data import *
import pickle
import datetime
import pickle
import gc
import copy
import defs
import sys
from multiprocessing import Pool

with open("test.txt", "rb") as fp:   # Unpickling
    context = pickle.load(fp)

with open("cnn8192.txt", "rb") as fp:   # Unpickling
    cnn_weight = pickle.load(fp)

with open("dense100_8192.txt", "rb") as fp:   # Unpickling
    dense_to_100_matrix = pickle.load(fp)

with open("mask8192.txt", "rb") as fp:   # Unpickling
    mask_plain = pickle.load(fp)

with open("dense10_8192.txt", "rb") as fp:   # Unpickling
    weight_10_plain = pickle.load(fp)

with open("bias100_8192.txt", "rb") as fp:   # Unpickling
    bias_100_plain = pickle.load(fp)

with open("encrypt8192.txt", "rb") as fp:   # Unpickling
    ct = pickle.load(fp)

pool = Pool(processes=64)

def CNN(ct):
    work = []
    for t in range(5):
        for i in range(25):
            test = {}
            test[0] = ct[i]
            test[1] = cnn_weight[t][i]
            work.append( test )
    result = pool.map(defs.plainmul, work)

    m_cnn = []
    for t in range(5):
        temp = []
        for i in range(25):
            temp.append(result[t*25+i])
        m_cnn.append(temp)

    result = pool.map(defs.add_to_first, m_cnn)

    work = []
    for i in range(1, 5):
        test = {}
        test[0] = result[i]
        test[1] = i
        work.append( test )
    rotated = pool.map(defs.rotate, work)

    for i in range(4):
        temp = result[0]
        result[0] = context.HAdd(rotated[i], temp)

    return result[0]

def Dense100(ct):
    # copy 8 times
    rt_2 = context.rotate_column(ct,   1024)
    rt_2 = context.HAdd(rt_2, ct)
    rt_4 = context.rotate_column(rt_2, 2048)
    rt_4 = context.HAdd(rt_2, rt_4)
    rt_8 = context.rotate_row(rt_4, 1)
    rt_8 = context.HAdd(rt_4, rt_8)

    work = []
    for i in range(13):
        test = {}
        test[0] = dense_to_100_matrix[i]
        test[1] = rt_8
        test[2] = i
        test[3] = mask_plain
        work.append(test)

    temp_result = pool.map(defs.dense100, work)
    
    for i in range(1, 13):
        temp = temp_result[0]
        temp_result[0] = context.HAdd(temp, temp_result[i])

    return context.PlainAdd(temp_result[0], bias_100_plain);

def Square(ct):
    return context.HMul(ct, ct)

def Dense10(ct):
    rt_2  = context.rotate_column(ct,   16)
    rt_2  = context.HAdd(rt_2, ct)
    rt_4  = context.rotate_column(rt_2, 32)
    rt_4  = context.HAdd(rt_2, rt_4)
    rt_8  = context.rotate_column(rt_4, 64)
    rt_8  = context.HAdd(rt_4, rt_8)
    rt_16 = context.rotate_column(rt_8, 128)
    rt_16 = context.HAdd(rt_8, rt_16)
    temp  = context.PlainMul(rt_16, weight_10_plain)
    temp2 = context.rotate_row(temp, 1)
    temp  = context.HAdd(temp, temp2)
    temp2 = context.rotate_column(temp, -2048)
    temp  = context.HAdd(temp, temp2)
    temp2 = context.rotate_column(temp, -1024)
    temp  = context.HAdd(temp, temp2)
    temp2 = context.rotate_column(temp, -8)
    temp  = context.HAdd(temp, temp2)
    temp2 = context.rotate_column(temp, -4)
    temp  = context.HAdd(temp, temp2)
    temp2 = context.rotate_column(temp, -2)
    temp  = context.HAdd(temp, temp2)
    temp2 = context.rotate_column(temp, -1)
    temp  = context.HAdd(temp, temp2)
    return temp

def Inference(ct):
    """
    print("cnn start: ", datetime.datetime.now(), flush=True)
    ct  = CNN(ct)
    print("cnn end: ", datetime.datetime.now(), flush=True)
    ct  = Square(ct)
    print("square end: ", datetime.datetime.now(), flush=True)
    with open("afterSquare_8192.txt", "wb") as fp:
        pickle.dump(ct, fp)
    ct  = Dense100(ct)
    with open("afterDense100_8192.txt", "wb") as fp:
        pickle.dump(ct, fp)
    print("dense100 end: ", datetime.datetime.now(), flush=True)
    """
    with open("afterDense100_8192.txt", "rb") as fp:   # Unpickling
        ct = pickle.load(fp)
    ct  = Square(ct)
    print("Square end: ", datetime.datetime.now(), flush=True)
    ct  = Dense10(ct)
    print("Dense10 end: ", datetime.datetime.now(), flush=True)
    pt  = context.decrypt(ct)
    print("decrypt end: ", datetime.datetime.now(), flush=True)
    out = context.decode_and_reconstruct(pt)
    print("decode end: ", datetime.datetime.now(), flush=True)

    result = []
    for i in range(10):
        result.append(out[16*i])
    print(result)
    print(result.index(max(result)))
    print([i/(1<<51) for i in out], flush=True)

Inference(ct)
"""
cnn_dump = []
for i in range(5):
    temp1 = []
    for j in range(25):
        temp1.append(int(conv1[i][j]*(1 << 5)))
    cnn_dump.append(temp1)
print("static const int cnn_weight[5][25] = " + (str(cnn_dump).replace("[", "{")).replace("]", "}") + ";")

dense100_dump = []
for i in range(13):
    temp1 = []
    for j in range(4):
        temp2 = []
        for k in range(7):
            temp3 = []
            for l in range(n):
                temp3.append( dense_to_100_matrix[i][j][k].poly[l] )
            temp2.append(temp3)
        temp1.append(temp2)
    dense100_dump.append(temp1)
print("static const int dense100[13][4][7][8192] = " + (str(dense100_dump).replace("[", "{")).replace("]", "}") + ";")

mask_plain_dump = []
for i in range(4):
    temp1 = []
    for j in range(7):
        temp2 = []
        for k in range(n):
            temp2.append(mask_plain[i][j].poly[k])
        temp1.append(temp2)
    mask_plain_dump.append(temp1)
print("static const int mask[4][7][8192] = " + (str(mask_plain_dump).replace("[", "{")).replace("]", "}") + ";")

weight_10_plain_dump = []
for i in range(4):
    temp1 = []
    for j in range(7):
        temp2 = []
        for k in range(n):
            temp2.append(weight_10_plain[i][j].poly[k])
        temp1.append(temp2)
    weight_10_plain_dump.append(temp1)
print("static const int dense10[4][7][8192] = " + (str(weight_10_plain_dump).replace("[", "{")).replace("]", "}") + ";")

bias_100_dump = []
for i in range(4):
    temp1 = []
    for j in range(7):
        temp2 = []
        for k in range(n):
            temp2.append(bias_100_plain[i][j].poly[k])
        temp1.append(temp2)
    bias_100_dump.append(temp1)
print("static const int bias100[4][7][8192] = " + (str(bias_100_dump).replace("[", "{")).replace("]", "}") + ";")

"""
