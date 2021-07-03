from bfv_rns import *
from data import *
def crt_mul(temp_plain):
    temp_plain = context.crt_and_encode(temp_plain)
    temp_plain = context.rns_ntt_pt_multiplication(temp_plain)
    return temp_plain

def crt_enc(temp_plain):
    temp_plain = context.crt_and_encode(temp_plain)
    temp_plain = context.encrypt(temp_plain)
    return temp_plain

def plainmul(input_):
    ct = input_[0]
    pt = input_[1]
    ret = context.PlainMul(ct, pt)
    return ret

def rotate(input_):
    temp = input_[0]
    i = input_[1]
    ret = context.rotate_column(temp, 169*i)
    return ret

def add_to_first(input_):
    temp = input_[0]
    for i in range(1, len(input_)):
        temp = context.HAdd(temp, input_[i])
    return temp

def dense100(input_):
    d = input_[0]        
    r = input_[1]
    i = input_[2]
    m = input_[3]
    temp = context.PlainMul(r, d)
    for j in range(9, -1, -1):
        temp2 = context.rotate_column(temp, -(1 << j))
        temp = context.HAdd(temp, temp2)
    temp = context.PlainMul(temp, m)
    temp = context.rotate_column(temp, i)
    return temp


with open("test.txt", "rb") as fp:   # Unpickling
    context = pickle.load(fp)
