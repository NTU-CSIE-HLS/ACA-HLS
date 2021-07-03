from bfv_rns import *
from data import *
import pickle
import datetime
import pickle
import gc
import copy
import defs
from multiprocessing import Pool

pool = Pool(processes=64)

def model_weight():
    work = []
    print("start mul: ", datetime.datetime.now())
    for c in range(5): 
        for i in range(25): 
            temp = int(conv1[c][i]*(1 << 5))
            temp_plain = [temp]*n
            work.append(temp_plain)

    for i in range(12):
        temp_plain = [0]*n
        for j in range(8):
            for k in range(845):
                temp = int(fc1[i*8+j][k]*(1 << 5))
                temp_plain[j*1024 + k] = temp
        work.append(temp_plain)
    temp_plain = [0]*n
    for j in range(4):
        for k in range(845):
            temp = int(fc1[12*8+j][k]*(1 << 5))
            temp_plain[j*1024 + k] = temp
    work.append(temp_plain)

    # int [4][7][8192]
    temp_plain = [0]*n
    for i in range(0, n, 1024):
        temp_plain[i] = 1
    work.append(temp_plain)

    temp_plain = [0]*n
    for i in range(10):
        for j in range(100):
            temp = int(fc2[i][j]*(1 << 5))
            temp_plain[((j*1024)//n) + (j*1024)%n + 16*i] = temp
    work.append(temp_plain)

    result = pool.map(defs.crt_mul, work)
    with open("result.txt", "wb") as fp:
        pickle.dump(result, fp)

    # int [5][25][4][7][8192]
    cnn_weight = []
    # int [13][4][7][8192]
    dense_to_100_matrix = []
    # int [4][7][8192]
    mask_plain = []
    # int [4][7][8192]
    weight_10_plain = []

    for c in range(5):
        temp = []
        for i in range(25): 
            temp.append(result[c*25+i])
        cnn_weight.append(temp)
    with open("cnn8192.txt", "wb") as fp:
        pickle.dump(cnn_weight, fp)
    for i in range(125, 125+13):
        dense_to_100_matrix.append(result[i])
    with open("dense100_8192.txt", "wb") as fp:
        pickle.dump(dense_to_100_matrix, fp)
    mask_plain = (result[125+13])
    with open("mask8192.txt", "wb") as fp:
        pickle.dump(mask_plain, fp)
    weight_10_plain = (result[125+14])
    with open("dense10_8192.txt", "wb") as fp:
        pickle.dump(weight_10_plain, fp)
    print(len(result))

    print("start bias100: ", datetime.datetime.now())
    # int [4][7][8192]
    bias_100_plain = [0]*n
    for i in range(100):
        temp = int(fc1_bias[i]*(1 << 23))
        bias_100_plain[((i*1024)//n) + (i*1024)%n] = temp
    bias_100_plain = context.crt_and_encode(bias_100_plain)
    bias_100_plain = context.rns_ntt_pt_addition(bias_100_plain)
    with open("bias100_8192.txt", "wb") as fp:
        pickle.dump(bias_100_plain, fp)

    return (cnn_weight, dense_to_100_matrix, mask_plain, bias_100_plain, weight_10_plain)

def encrypt_image(data):
    # int [25][4][7][8192]
    encrypted_matrix = []
    work = []
    for p in range(5):
        for q in range(5):
            k = 0
            which = p*5+q
            temp_plain = [0]*n
            ii = p
            for i in range(13):
                jj = q
                for j in range(13):
                    temp = int(data[ii][jj]*(1 << 4))
                    temp_plain[k] = temp
                    k += 1
                    jj += 2
                ii += 2
            work.append(temp_plain)
    result = pool.map(defs.crt_enc, work)
    with open("encrypt8192.txt", "wb") as fp:
        pickle.dump(result, fp)
    for i in range(25):
        encrypted_matrix.append(result[i])
    return encrypted_matrix

def CNN(ct):
    m_cnn = []
    for t in range(5):
        ttt = []
        for i in range(25):
            temp = context.PlainMul(ct[i], cnn_weight[t][i])
            ttt.append(temp)
        m_cnn.append(ttt)

    result = []
    for t in range(5):
        temp = m_cnn[t][0]
        for i in range(1, 25):
            temp = context.HAdd(temp, m_cnn[t][i])
        result.append(temp)

    result_rt = []
    for i in range(1, 5):
        temp = m_cnn[0][0]
        m_cnn[i][0] = context.rotate_column(temp, 169*i)

    for i in range(1, 5):
        temp = m_cnn[0][0]
        m_cnn[0][0] = context.HAdd(m_cnn[i][0], temp)

    return m_cnn[0][0]

def Square(ct):
    return context.HMul(ct, ct)

def Dense100(ct):
    # copy 8 times
    rt_2 = context.rotate_column(ct,   1024)
    rt_2 = context.HAdd(rt_2, ct)
    rt_4 = context.rotate_column(rt_2, 2048)
    rt_4 = context.HAdd(rt_2, rt_4)
    rt_8 = context.rotate_row(rt_4, 1)
    rt_8 = context.HAdd(rt_4, rt_8)

    temp = [None]*13
    for i in range(13):
        if i == 12:
            temp[i] = context.PlainMul(dense_to_100_matrix[i], rt_4)
        else:
            temp[i] = context.PlainMul(dense_to_100_matrix[i], rt_8)
        temp2 = temp[i]
        for j in range(9, -1, -1):
            temp2 = context.rotate_column(temp[i], -(1 << j))
            temp[i] = context.HAdd(temp[i], temp2)
        temp[i] = context.HAdd(temp[i], mask_plain)
        temp[i] = context.rotate_column(temp[i], i)
    
    for i in range(1, 13):
        temp[0] = context.HAdd(temp[0], temp[i])

    return context.HAdd(temp[0], bias_100_plain);

def Dense10(ct):
    rt_2  = context.rotate_column(ct,   16)
    rt_2  = context.HAdd(rt_2, ct)
    rt_4  = context.rotate_column(rt_2, 32)
    rt_4  = context.HAdd(rt_2, rt_4)
    rt_8  = context.rotate_column(rt_4, 64)
    rt_8  = context.HAdd(rt_4, rt_8)
    rt_16 = context.rotate_column(rt_8, 128)
    rt_16 = context.HAdd(rt_8, rt_16)
    temp  = context.HMul(rt_16, weight_10_plain)
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
    ct  = CNN(ct)
    # ct  = Square(ct)
    # ct  = Dense100(ct)
    # ct  = Square(ct)
    # ct  = Dense10(ct)
    pt  = context.decrypt(ct)
    out = context.decode_and_reconstruct(pt)

    # result = []
    # for i in range(10):
    #     result.append(out[16*i])
    # print(result)
    # print(result.index(max(result)))
    print([i/(1<<20) for i in out])


# w = not_model()
# p = not_encrypt_image(test_data)
# CCC(p, w)

print(datetime.datetime.now())
with open("test.txt", "rb") as fp:   # Unpickling
    context = pickle.load(fp)
#print(datetime.datetime.now())
#parms = model_weight()
#with open("model8192.txt", "wb") as fp:
#    pickle.dump(parms, fp)
# with open("model1024.txt", "rb") as fp:   # Unpickling
#     parms = pickle.load(fp)
# cnn_weight = parms
print("start enc: ", datetime.datetime.now())
ct = encrypt_image(test_data)
with open("encrypt8192.txt", "wb") as fp:
    pickle.dump(ct, fp)
# with open("encrypt1024.txt", "rb") as fp:   # Unpickling
#     ct = pickle.load(fp)

#print(datetime.datetime.now())
#Inference(ct)
#print(datetime.datetime.now())
