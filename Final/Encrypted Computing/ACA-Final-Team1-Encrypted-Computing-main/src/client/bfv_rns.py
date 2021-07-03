from sympy import *
from decimal import *
import random
import sys
import math
import datetime
import pickle

getcontext().prec = 300

def center(a, b):
    if a > b//2:
        a -= b
    elif a < -b//2:
        a += b
    return a

def modInverse(a, m): 
    m0 = m 
    y = 0
    x = 1 
    if (m == 1) : 
        return 0
    while (a > 1) : 
        q = a // m 
        t = m 
        m = a % m 
        a = t 
        t = y 
        y = x - q * y 
        x = t 
    if (x < 0) : 
        x = x + m0  
    return center(x, m)

def montgomery_mul(a, b, MOD, Mprime, R):
    # b is table
    temp = a * b
    lower = (temp % R)
    tmp = (lower * Mprime) % R
    out = tmp * MOD + temp
    shift_num = int(math.log(R, 2))
    out = out >> shift_num
    if out > MOD: # range [0, 2*MOD]
        out -= MOD
    return out

def FindPrime(q, n):
    # find prime
    while True:
        q = nextprime(q)
        if (q - 1) % (2*n) == 0:
            break
    return q

def sample(num, q, n): # R_num
    ret = [0]*n
    for i in range(n):
        ret[i] = random.randint(0,num-1)
    return polynomial(ret, q)

def bit_decomposition(a, q, T):
    l = math.floor(math.log(q, T))
    out = [[0 for x in range(a.n)] for y in range((l+1))]
    for i in range(a.n):    
        tmp = a.poly[i]
        for j in range(l+1):
            num = tmp % T
            out[j][i] = num
            tmp -= num
            tmp = tmp // T
    ret = []
    for i in range(l+1):
        ret.append(polynomial(out[i], q))
    return ret

def bit_combination(polys, q, T):
    l = math.floor(math.log(q, T))
    out = [0]*polys[0].n
    for i in range(polys[0].n):    
        tmp = 0
        for j in range(l+1):
            tmp = tmp + (polys[j].poly[i]*T**j)
        out[i] = tmp
    return polynomial(ret, mod, False)

def ntt_inverse(poly, table, Mprime):
    ret = [i for i in poly.poly]
    mod = poly.mod
    for i in range(int(math.log(poly.n, 2))):
        step = 2**(i)
        group = int(poly.n // len(table[int(math.log(poly.n, 2))-1-i]))
        for k in range(len(table[int(math.log(poly.n, 2))-1-i])):
            for j in range(step):
                ta = ret[k*group + j]
                tb = ret[k*group + j + step]
                ret[k*group + j]        = center((ta + tb) % mod, mod)
                ret[k*group + j + step] = center(
                    montgomery_mul((ta - tb),table[int(math.log(n, 2))-1-i][k], mod, Mprime, 2**32), mod)
    for i in range(poly.n):
        ret[i] = center(ret[i] * modInverse(poly.n, mod) % mod, mod)
    return polynomial(ret, mod, False)

def ntt_forward(poly, table, Mprime):
    ret = [i for i in poly.poly]
    mod = poly.mod
    for i in range(int(math.log(poly.n, 2))):
        step = 2**(int(math.log(poly.n, 2))-1-i)
        group = int(poly.n // len(table[i]))
        for k in range(len(table[i])):
            for j in range(step):
                ta = ret[k*group + j]
                tb = ret[k*group + j + step]
                ret[k*group + j]        = center((ta + montgomery_mul(tb, table[i][k], mod, Mprime, 2**32)) % mod, mod)
                ret[k*group + j + step] = center((ta - montgomery_mul(tb, table[i][k], mod, Mprime, 2**32)) % mod, mod)
    return polynomial(ret, mod, True)

def ntt_inverse_bunch(polys, tables, Mprimes):
    ret = []
    for poly, table, Mprime in zip(polys, tables, Mprimes):
        ret.append(ntt_inverse(poly, table, Mprime))
    return ret

def ntt_forward_bunch(polys, tables, Mprimes):
    ret = []
    for poly, table, Mprime in zip(polys, tables, Mprimes):
        ret.append(ntt_forward(poly, table, Mprime))
    return ret

# R_q
class polynomial(object):
    def __init__(self, input_list, mod, is_ntt = False):
        super(polynomial, self).__init__()
        if isinstance(input_list, list):
            self.poly = [center(i, mod) for i in input_list]
            self.n = len(self.poly)
            self.mod = mod
            self.ntt = is_ntt
        else:
            raise Exception("Polynomial input should be list.")

    def __str__(self):
        return str(self.poly)

    def __mul__(self, other):
        if isinstance(other, polynomial) and not self.ntt and not other.ntt:
            if self.mod != other.mod or self.n != other.n:
                raise Exception("mod or n not the same")
            temp = [0] * (2*self.n)
            for i in range(self.n):
                for j in range(self.n):
                    temp[i+j] = (temp[i+j] + self.poly[i] * other.poly[j]) % self.mod
            ret = [0] * self.n
            for i in range(self.n):
                ret[i] = center((temp[i] - temp[i+self.n]) % self.mod, self.mod)
            return self.__class__(ret, self.mod, self.ntt)
        elif isinstance(other, polynomial) and self.ntt and other.ntt:
            if self.mod != other.mod or self.n != other.n:
                raise Exception("mod or n not the same")
            ret = [0] * self.n
            for i in range(self.n):
                ret[i] = center((self.poly[i] * other.poly[i]) % self.mod, self.mod)
            return self.__class__(ret, self.mod, self.ntt)
        elif (isinstance(other, int) or isinstance(other, float)) and not self.ntt:
            temp = [0] * self.n
            for i in range(self.n):
                temp[i] = center((self.poly[i] * other) % self.mod, self.mod)
            return self.__class__(temp, self.mod)
        else:
            raise Exception("Not supported")

    def __add__(self, other):
        if isinstance(other, polynomial):
            if self.mod != other.mod or self.n != other.n:
                raise Exception("mod or n not the same")
            temp = [0] * self.n
            for i in range(self.n):
                temp[i] = center((self.poly[i] + other.poly[i]) % self.mod, self.mod)
            return self.__class__(temp, self.mod, self.ntt)
        elif isinstance(other, int):
            temp = [0] * self.n
            for i in range(self.n):
                temp[i] = center((self.poly[i] + other) % self.mod, self.mod)
            return self.__class__(temp, self.mod, self.ntt)
        else:
            raise Exception("Not supported")

    def __neg__(self):
        temp = [0] * self.n
        for i in range(self.n):
            temp[i] = center((-self.poly[i]) % self.mod, self.mod)
        return self.__class__(temp, self.mod)

    def change_mod(self, new_mod):
        if self.mod > new_mod:
            return polynomial([center(self.poly[i] % new_mod, new_mod) for i in range(self.n)], new_mod)
        return polynomial(self.poly, new_mod)

    def center(self):
        for i in range(self.n):
            self.poly[i] = center(self.poly[i], self.mod)

class BFVEncryptor(object):
    def __init__(self, q, p, n, t, q_i, p_i, q_tilde, p_tilde, q_star, p_star, Q_tilde_q, Q_tilde_p,
            ntt_forward_table, ntt_inverse_table, Mprimes):
        super(BFVEncryptor, self).__init__()
        
        self.q_i = q_i
        self.p_i = p_i
        self.q_tilde = q_tilde
        self.p_tilde = p_tilde
        self.q_star = q_star
        self.p_star = p_star
        self.Q_tilde_q = Q_tilde_q
        self.Q_tilde_p = Q_tilde_p
        self.ntt_forward_table = ntt_forward_table
        self.ntt_inverse_table = ntt_inverse_table
        self.Mprimes = Mprimes
        self.RNS_NUM = len(self.q_i) + 1
        self.m = n*2
        self.n = n
        self.bits = int(math.log(self.n, 2))
        self.q = q
        self.p = p
        self.t = t
        self.Q = self.p*self.q
        self.delta = self.q // self.t
        self.R = 2**32

        self.keygen()
        self.evaluation_keygen()
        self.galois_keygen()

    def keygen(self):
        a = sample(self.q, self.q, self.n)
        s = sample(     2, self.q, self.n)
        e = sample(     2, self.q, self.n)
        self.pk = (-(a*s+e), a)
        self.sk = s

    def evaluation_keygen(self):
        self.rlk = []
        s_square = self.sk*self.sk

        for i, q_i in enumerate(self.q_i):
            a = sample(self.q, self.q, self.n)
            e = sample(     2, self.q, self.n)
            (alpha, beta) = (-(a*self.sk)+e+s_square*self.q_star[i]*self.q_tilde[i], a)
            rlk_0 = []; rlk_1 = []
            for j, q_j in enumerate(self.q_i):
                rlk_0.append( ntt_forward(alpha.change_mod(q_j), self.ntt_forward_table[j], self.Mprimes[j]) )
                rlk_1.append( ntt_forward(beta.change_mod(q_j),  self.ntt_forward_table[j], self.Mprimes[j]) )
            self.rlk.append( (rlk_0, rlk_1) )

    def galois_keygen(self):
        self.galois = []
        
        for sign in [1, -1]:
            for p in [1<<pp for pp in range(self.bits-1)]:
                temp = []
                s_ = self.rotate(self.sk, (p*sign + (self.n//2)) % (self.n//2), 3)
                for i, q_i in enumerate(self.q_i):
                    a = sample(self.q, self.q, self.n)
                    e = sample(     2, self.q, self.n)
                    (alpha, beta) = (-(a*self.sk)+e+s_*self.q_star[i]*self.q_tilde[i], a)
                    gk_0 = []; gk_1 = []
                    for j, q_j in enumerate(self.q_i):
                        gk_0.append( ntt_forward(alpha.change_mod(q_j), self.ntt_forward_table[j], self.Mprimes[j]) )
                        gk_1.append( ntt_forward(beta.change_mod(q_j),  self.ntt_forward_table[j], self.Mprimes[j]) )
                    temp.append( (gk_0, gk_1) )
                self.galois.append(temp)
        temp = []
        s_ = self.rotate(self.sk, 1, self.m-1)
        for i, q_i in enumerate(self.q_i):
            a = sample(self.q, self.q, self.n)
            e = sample(     2, self.q, self.n)
            (alpha, beta) = (-(a*self.sk)+e+s_*self.q_star[i]*self.q_tilde[i], a)
            gk_0 = []; gk_1 = []
            for j, q_j in enumerate(self.q_i):
                gk_0.append( ntt_forward(alpha.change_mod(q_j), self.ntt_forward_table[j], self.Mprimes[j]) )
                gk_1.append( ntt_forward(beta.change_mod(q_j),  self.ntt_forward_table[j], self.Mprimes[j]) )
            temp.append( (gk_0, gk_1) )
        self.galois.append(temp)

    # (delta*m + pk[0]*u + e1, pk[1]*u + e2)
    def encrypt(self, poly):
        e1 = sample(2, self.q, self.n)
        e2 = sample(2, self.q, self.n)
        u  = sample(2, self.q, self.n)
        m = poly.change_mod(self.q)
        tmp_ct = (m*self.delta + self.pk[0]*u + e1, self.pk[1]*u + e2)
        
        ct0 = []; ct1 = [];
        for i, q_i in enumerate(self.q_i):
            ct0.append( ntt_forward(tmp_ct[0].change_mod(q_i), self.ntt_forward_table[i], self.Mprimes[i]) )
            ct1.append( ntt_forward(tmp_ct[1].change_mod(q_i), self.ntt_forward_table[i], self.Mprimes[i]) )
        
        return (ct0, ct1)

    def decrypt(self, ct):     
        ct = (ntt_inverse_bunch(ct[0], self.ntt_inverse_table, self.Mprimes),
              ntt_inverse_bunch(ct[1], self.ntt_inverse_table, self.Mprimes))

        polys = []
        for i, mod in enumerate(self.q_i):
            x = (ct[0][i] + ct[1][i]*self.sk.change_mod(mod))
            polys.append(x.poly)
        ret = [0] * self.n
        for i in range(self.n):
            temp = 0
            for j, mod in enumerate(self.q_i):            
                temp += polys[j][i] * self.q_tilde[j] * Decimal(self.t) / Decimal(mod)
            ret[i] = int(round(temp)) % self.t
        
        return polynomial(ret, self.t)

    def printBudget(self, ct):
        ct = (ntt_inverse_bunch(ct[0], self.ntt_inverse_table, self.Mprimes), 
              ntt_inverse_bunch(ct[1], self.ntt_inverse_table, self.Mprimes))
        polys = []
        for i, mod in enumerate(self.q_i):
            x = (ct[0][i] + ct[1][i]*self.sk.change_mod(mod))
            polys.append(x.poly)
        first_element = 0
        for i, mod in enumerate(self.q_i):     
            first_element += polys[i][0] * self.q_tilde[i] * Decimal(self.t) / Decimal(mod)
        budget = abs(math.log(abs((first_element-round(first_element))), 2))
        if abs((first_element-round(first_element))) > 0.5:
            print("Budget: ", 0)
        else:
            print("Budget: ", budget)

    def PlainAdd(self, rns_a, plain):
        ct0 = []
        for a, b in zip(rns_a[0], plain):
            ct0.append(a + b)
        
        return (ct0, rns_a[1])

    def PlainMul(self, rns_a, plain):
        ct0 = []; ct1 = []
        for (a, b, c) in zip(rns_a[0], rns_a[1], plain):
            # plain = plain.change_mod(a.mod) # decomposition
            ct0.append(a * c)
            ct1.append(b * c)
        
        return (ct0, ct1)

    def HAdd(self, rns_a, rns_b):
        ct0 = []; ct1 = []
        for i in range(len(self.q_i)):
            ct0.append(rns_a[0][i] + rns_b[0][i])
            ct1.append(rns_a[1][i] + rns_b[1][i])
        
        return (ct0, ct1)

    def HMul(self, rns_a, rns_b):
        # back to normal form
        inv_nttd_rns_a_0 = ntt_inverse_bunch(rns_a[0], self.ntt_inverse_table, self.Mprimes)
        inv_nttd_rns_a_1 = ntt_inverse_bunch(rns_a[1], self.ntt_inverse_table, self.Mprimes)
        inv_nttd_rns_b_0 = ntt_inverse_bunch(rns_b[0], self.ntt_inverse_table, self.Mprimes)
        inv_nttd_rns_b_1 = ntt_inverse_bunch(rns_b[1], self.ntt_inverse_table, self.Mprimes)

        # Basis extension Zq -> Zpq        
        rns_aa = (self.basis_extension_forward(inv_nttd_rns_a_0), 
                  self.basis_extension_forward(inv_nttd_rns_a_1) )
        rns_bb = (self.basis_extension_forward(inv_nttd_rns_b_0), 
                  self.basis_extension_forward(inv_nttd_rns_b_1) )
        
        # ntt forward
        nttd_rns_a_0 = ntt_forward_bunch(rns_aa[0], self.ntt_forward_table, self.Mprimes)
        nttd_rns_a_1 = ntt_forward_bunch(rns_aa[1], self.ntt_forward_table, self.Mprimes)
        nttd_rns_b_0 = ntt_forward_bunch(rns_bb[0], self.ntt_forward_table, self.Mprimes)
        nttd_rns_b_1 = ntt_forward_bunch(rns_bb[1], self.ntt_forward_table, self.Mprimes)
        
        # Tensoring
        c0 = []; c1 = []; c2 = [];
        for i in range(len(self.q_i + self.p_i)):
            c0.append(nttd_rns_a_0[i]*nttd_rns_b_0[i])
            c1.append(nttd_rns_a_0[i]*nttd_rns_b_1[i] + nttd_rns_a_1[i]*nttd_rns_b_0[i])
            c2.append(nttd_rns_a_1[i]*nttd_rns_b_1[i])
        
        # ntt inverse
        c0 = ntt_inverse_bunch(c0, self.ntt_inverse_table, self.Mprimes)
        c1 = ntt_inverse_bunch(c1, self.ntt_inverse_table, self.Mprimes)
        c2 = ntt_inverse_bunch(c2, self.ntt_inverse_table, self.Mprimes)
        
        # Scale down t/q x, Zpq -> Zp
        c0 = self.scaling(c0)
        c1 = self.scaling(c1)
        c2 = self.scaling(c2)
        
        # Extension back, Zp -> Zq
        c0 = self.basis_extension_backward(c0)
        c1 = self.basis_extension_backward(c1)
        c2 = self.basis_extension_backward(c2)
        
        return self.relin(c0, c1, c2)
    
    def scaling(self, a):
        # a dimension: (rns, polynomial)
        tmp = []
        for i, ii in enumerate(self.p_i):
            poly = []
            for j in range(self.n):
                temp = 0
                for idx, k in enumerate(self.q_i):
                    temp += (a[idx].poly[j] * (self.t * self.Q_tilde_q[idx] * self.p / Decimal(k)))
                temp = int(round(temp))
                temp = (temp + (a[len(self.q_i)+i].poly[j] * ((self.t*self.Q_tilde_p[i]*self.p_star[i])) % ii) % ii) % ii
                poly.append(center(temp, ii))
            tmp.append(polynomial(poly, ii))
        
        return tmp

    def basis_extension_forward(self, a): # q -> pq
        # a dimension: (rns, polynomial)
        tmp = []
        for i in self.p_i:
            poly1 = []
            for j in range(self.n):
                temp1 = 0
                v1 = 0
                pre1 = []
                for idx, k in enumerate(self.q_i):
                    pre1.append(center(a[idx].poly[j]*self.q_tilde[idx], k))
                for idx, k in enumerate(self.q_i):
                    temp1 = (temp1 + (pre1[idx]*(self.q_star[idx] % i)) % i) % i
                    v1 += (pre1[idx])/Decimal(k)
                v1 = (int(round(v1)) * (self.q % i)) % i
                temp1 = center(temp1 - v1, i)
                poly1.append(temp1)
            tmp.append( polynomial(poly1, i) )
        
        return a + tmp

    def basis_extension_backward(self, a): # p -> q
        # a dimension: (rns, polynomial)
        tmp = []
        for i in self.q_i:
            poly1 = []
            for j in range(self.n):
                temp1 = 0
                v1 = 0
                pre1 = []
                for idx, k in enumerate(self.p_i):
                    pre1.append(center(a[idx].poly[j]*self.p_tilde[idx], k))
                for idx, k in enumerate(self.p_i):
                    temp1 = (temp1 + (pre1[idx]*(self.p_star[idx] % i)) % i) % i
                    v1 += (pre1[idx])/Decimal(k)
                v1 = (int(round(v1)) * (self.p % i)) % i
                if temp1 > v1:
                    temp1 = temp1 - v1
                else:
                    temp1 = temp1 + i - v1
                poly1.append(temp1)
            tmp.append( polynomial(poly1, i) )
        
        return tmp

    def relin(self, c0, c1, c2):
        nttd_c0 = []; nttd_c1 = []
        for i in range(len(self.q_i)):
            nttd_c0.append( ntt_forward(c0[i], self.ntt_forward_table[i], self.Mprimes[i]) )
            nttd_c1.append( ntt_forward(c1[i], self.ntt_forward_table[i], self.Mprimes[i]) )
        decomposition = self.keyswitching(c2, self.rlk)
        
        return self.HAdd((nttd_c0, nttd_c1), decomposition)

    def rotate(self, poly, r, basis):
        rotate_in_Z_m = basis**r
        after0 = [0]*self.n

        for i in range(self.n):
            new_value = poly.poly[i]
            if ((i*rotate_in_Z_m) >> self.bits) % 2 == 1: # multiply by minus one
                new_value *= -1
            new_idx = (i*rotate_in_Z_m) % self.n
            after0[new_idx] = new_value

        return polynomial(after0, poly.mod)

    def rotate_column(self, poly, r, idx):
        r = (r+(self.n//2))%(self.n//2)
        ct0 = []
        ct1 = []
        for i in range(len(self.q_i)):
            a = self.rotate(ntt_inverse(poly[0][i], self.ntt_inverse_table[i], self.Mprimes[i]), r, 3)
            b = self.rotate(ntt_inverse(poly[1][i], self.ntt_inverse_table[i], self.Mprimes[i]), r, 3)
            ct0.append(ntt_forward(a, self.ntt_forward_table[i], self.Mprimes[i]))
            ct1.append( b )
        
        decomposition = self.keyswitching(ct1, self.galois[idx])
        
        ret = []
        for i in range(len(self.q_i)):
            ret.append(ct0[i] + decomposition[0][i])

        return (ret, decomposition[1])

    def rotate_row(self, poly, r, idx):
        r = (r+2)%2
        ct0 = []
        ct1 = []
        for i in range(len(self.q_i)):
            a = self.rotate(ntt_inverse(poly[0][i], self.ntt_inverse_table[i], self.Mprimes[i]), r, self.m-1)
            b = self.rotate(ntt_inverse(poly[1][i], self.ntt_inverse_table[i], self.Mprimes[i]), r, self.m-1)
            ct0.append(ntt_forward(a, self.ntt_forward_table[i], self.Mprimes[i]))
            ct1.append( b )
        
        decomposition = self.keyswitching(ct1, self.galois[idx])

        ret = []
        for i in range(len(self.q_i)):
            ret.append(ct0[i] + decomposition[0][i])

        return (ret, decomposition[1])

    def keyswitching(self, rns, keys):
        ct0 = []
        ct1 = []
        for i, q_i in enumerate(self.q_i):
            c0_ = 0
            c1_ = 0             
            for j in range(len(self.q_i)):
                decomposed_c1 = rns[j].change_mod(q_i) # basis decomposition
                decomposed_c1 = ntt_forward(decomposed_c1, self.ntt_forward_table[i], self.Mprimes[i])
                c0_ = (decomposed_c1 * keys[j][0][i]) + c0_
                c1_ = (decomposed_c1 * keys[j][1][i]) + c1_
            ct0.append( c0_ )
            ct1.append( c1_ )
        return (ct0, ct1)

class BFVEncoder(object):
    def __init__(self, t, n):
        super(BFVEncoder, self).__init__()
        
        self.n = n
        self.m = self.n*2
        self.t = t
        ### find generator
        G = 2
        for i in range(2, self.t):
            if is_primitive_root(i, self.t):
                G = i
                break
        self.G = G

        self.root = pow(self.G, ((self.t-1)//self.m), self.t)
        generators = [3, self.m-1]
        orders = [n_order(i, self.m) for i in generators]

        basis = [0]*self.n
        for i in range(self.n):
            basis[i] = generators[0]**(i%orders[0]) * generators[1]**(i//orders[0]) % self.m

        self.basis = basis
        #print(self.basis, self.root)

    def encode(self, poly):
        ret = [0] * self.n
        for i in range(self.n):
            s = 0
            for j in range(self.n):
                s = (s + poly[j]*pow(self.root, (-i*self.basis[j]), self.t)) % self.t
            ret[i] = s
        for i in range(self.n):
            ret[i] = ret[i] * modInverse(self.n, self.t) % self.t
        return polynomial(ret, self.t)

    def decode(self, poly):
        ret = [0] * self.n
        for i in range(self.n):
            s = 0
            for j in range(self.n):
                s = (s + poly.poly[j]*pow(self.root, (j*self.basis[i]), self.t)) % self.t
            ret[i] = center(s, self.t)
        return ret

    def rns_ntt_pt_addition(self, poly, encryptor):
        pt = []
        for q_i in encryptor.q_i:
            temp = encryptor.delta % q_i
            pt.append(poly.change_mod(q_i) * temp)
        return ntt_forward_bunch(pt, encryptor.ntt_forward_table, encryptor.Mprimes)

    def rns_ntt_pt_multiplication(self, poly, encryptor):
        pt = []
        for q_i in encryptor.q_i:
            pt.append(poly.change_mod(q_i))
        return ntt_forward_bunch(pt, encryptor.ntt_forward_table, encryptor.Mprimes)

class BFVContext(object):
    def __init__(self, bit_list, n, t):
        super(BFVContext, self).__init__()

        security = {1024: 27, 2048: 54, 4096: 109, 8192: 218, 16384: 438, 32768: 881}
        total_bits = 0
        for i in bit_list:
            total_bits += i
        if total_bits > security[n]:
            raise Exception("Securtiy check failed")

        self.t = t
        self.n = n
        self.t_all = 1

        # Find primes
        self.q_i = []
        q = 1
        for i in bit_list:
            prime = FindPrime(2**i, n)
            while prime in self.q_i:
                prime = FindPrime(prime, n)
            self.q_i.append(prime)
            q *= prime
        self.p_i = []
        p = 1
        for i in bit_list + [bit_list[-1]]:
            prime = FindPrime(2**i, n)
            while prime in self.q_i or prime in self.p_i:
                prime = FindPrime(prime, n)
            self.p_i.append(prime)
            p *= prime
       
        self.RNS_NUM = len(self.q_i) + 1
        self.m = n*2
        self.n = n
        self.bits = int(math.log(self.n, 2))
        self.q = q
        self.p = p
        self.R = 2**32
        self.Q = self.p*self.q
        
        self.set_constants() 

        self.encryptor = []
        self.encoder   = []
        
        for i in self.t:
            self.encoder.append(BFVEncoder(i, self.n))
            self.encryptor.append(BFVEncryptor(self.q, self.p, self.n, i, self.q_i, self.p_i, self.q_tilde, self.p_tilde, 
                self.q_star, self.p_star, self.Q_tilde_q, self.Q_tilde_p,
                self.ntt_forward_table, self.ntt_inverse_table, self.Mprimes))
            self.t_all *= i

        self.t_star  = []
        self.t_tilde = []
        for i in self.t:
            temp = int(self.t_all // i)
            self.t_star.append( temp )
            self.t_tilde.append( center(modInverse(temp, i), i) )

    def rns_ntt_pt_addition(self, pt):
        ret = []
        for p, encoder, encryptor in zip(pt, self.encoder, self.encryptor):
            ret.append( encoder.rns_ntt_pt_addition(p, encryptor) )
        return ret

    def rns_ntt_pt_multiplication(self, pt):
        ret = []
        for p, encoder, encryptor in zip(pt, self.encoder, self.encryptor):
            ret.append( encoder.rns_ntt_pt_multiplication(p, encryptor) )
        return ret

    def encrypt(self, pt):
        ret = []
        for p, encryptor in zip(pt, self.encryptor):
            ret.append(encryptor.encrypt(p))
        return ret

    def decrypt(self, ct):
        ret = []
        for c, encryptor in zip(ct, self.encryptor):
            ret.append(encryptor.decrypt(c))
        return ret

    def PlainAdd(self, ct, pt):
        ret = []
        for c, p, encryptor in zip(ct, pt, self.encryptor):
            ret.append( encryptor.PlainAdd(c, p) )
        return ret

    def PlainMul(self, ct, pt):
        ret = []
        for c, p, encryptor in zip(ct, pt, self.encryptor):
            ret.append( encryptor.PlainMul(c, p) )
        return ret

    def HAdd(self, ct0, ct1):
        ret = []
        for c0, c1, encryptor in zip(ct0, ct1, self.encryptor):
            ret.append(encryptor.HAdd(c0, c1))
        return ret

    def HMul(self, ct0, ct1):
        ret = []
        for c0, c1, encryptor in zip(ct0, ct1, self.encryptor):
            ret.append(encryptor.HMul(c0, c1))
        return ret

    def decompose_rotate(self, a):
        l = []
        while a != 0:
            t = (round(math.log(abs(a), 2)))
            num = 1<<t
            if a < 0:
                num = -num
                t   = t + (int(math.log(self.n, 2))-1)
            a -= num
            if num % (self.n//2) == 0:
                continue
            l.append((num, t))
        return l

    def rotate_column(self, ct, r):
        r = -r
        if r >= self.n // 2 and r <= -self.n // 2:
            raise Exception("Not supported")
        
        l = self.decompose_rotate(r)
        print(l)
        ret = []
        for c, encryptor in zip(ct, self.encryptor):
            cc = c
            for i in l:
                cc = encryptor.rotate_column(cc, i[0], i[1])
            ret.append(cc)

        return ret

    def rotate_row(self, ct, r):
        if r != 1:
            raise Exception("Not supported")
        ret = []
        for c, encryptor in zip(ct, self.encryptor):
            ret.append(encryptor.rotate_row(c, r, 2*(self.bits-1)))
        ret.append(ret)
        return ret

    def decode_and_reconstruct(self, polys):
        decoded_poly = []
        for i, p in enumerate(polys):
            decoded_poly.append( self.encoder[i].decode(p) )

        ret = []
        for i in range(self.n):
            temp = 0
            for j, poly in enumerate(decoded_poly):
                temp += poly[i] * self.t_star[j] * self.t_tilde[j]
            ret.append( center(temp % self.t_all, self.t_all) )
        return ret

    def crt_and_encode(self, poly):
        ret = []
        for i, t in enumerate(self.t):
            temp = [0]*self.n
            for j in range(self.n):
                temp[j] = center(poly[j] % t, t)
            ret.append(self.encoder[i].encode(temp))
        return ret

    def printBudget(self, ct):
        self.encryptor[0].printBudget(ct[0])

    def set_constants(self):
        # strings
        mod_str        = "static const ap_uint<PRIME_BIT> MOD[2*RNS_NUM] = {"
        N_inv_str      = "static const ap_int<PRIME_BIT> N_inv[2*RNS_NUM] =  {"
        Mprimes_str    = "static const ap_int<PRIME_BIT> Mprime[2*RNS_NUM] =  {"
        one_over_q_str = "static const ap_uint<3*PRIME_BIT> one_over_q[2*RNS_NUM] = {" 
        q_tilde_str    = "static const ap_int<PRIME_BIT> q_tilde[2*RNS_NUM] = {"
        q_tilde_str_R  = "static const ap_int<PRIME_BIT> q_tilde[2*RNS_NUM] = {"
        q_p_mod_str    = "static const ap_int<PRIME_BIT> q_p[2*RNS_NUM] = {" 
        q_p_mod_str_R  = "static const ap_int<PRIME_BIT> q_p[2*RNS_NUM] = {" 
        q_to_p_str     = "static const ap_int<PRIME_BIT> q_to_p[RNS_NUM][RNS_NUM] = {"
        q_to_p_str_R   = "static const ap_int<PRIME_BIT> q_to_p[RNS_NUM][RNS_NUM] = {"
        p_to_q_str     = "static const ap_int<PRIME_BIT> p_to_q[RNS_NUM][RNS_NUM] = {"
        p_to_q_str_R   = "static const ap_int<PRIME_BIT> p_to_q[RNS_NUM][RNS_NUM] = {"
        Theta          = "static const ap_uint<2*PRIME_BIT> Theta[NUM_PLAIN][RNS_NUM-1] = {"
        Omega          = "static const ap_int<PRIME_BIT> Omega[NUM_PLAIN][RNS_NUM-1][RNS_NUM] = {"
        Omega_R        = "static const ap_int<PRIME_BIT> Omega[NUM_PLAIN][RNS_NUM-1][RNS_NUM] = {"
        tQ             = "static const ap_int<PRIME_BIT> tQ[NUM_PLAIN][RNS_NUM] = {"
        tQ_R           = "static const ap_int<PRIME_BIT> tQ[NUM_PLAIN][RNS_NUM] = {"
        forward        = "static const int forward_table[2*RNS_NUM-1][N] = {"
        inverse        = "static const int inv_table[2*RNS_NUM-1][N] = {"
        T_str          = "static const ap_int<PRIME_BIT> T[NUM_PLAIN] = {"
        delta_str      = "static const ap_int<PRIME_BIT> delta[NUM_PLAIN][RNS_NUM-1] = {"

        T_str   += (str(self.t)[1:-1] + "};")
        mod_str += (str(self.q_i)[1:-1] + ", 0, ")
        mod_str += (str(self.p_i)[1:-1] + "};")
        
        for i in self.t:
            temp = []
            for q_i in self.q_i:
                temp.append((self.q // i) % q_i)
            delta_str += ("{" + (str(temp)[1:-1] + "},"))
        delta_str += "};"
        
        self.Ninv    = []
        self.Mprimes = []
        for i in self.q_i:
            self.Ninv.append( center((modInverse(self.n, i) + i) % i, i) )
            self.Mprimes.append( center(modInverse(-i % self.R, self.R), self.R) )
        N_inv_str   += (str(self.Ninv)[1:-1] + ", 0, ")
        Mprimes_str += (str(self.Mprimes)[1:-1] + ", 0, ")
        for i in self.p_i:
            self.Ninv.append( center((modInverse(self.n, i) + i) % i, i) )
            self.Mprimes.append( center(modInverse(-i % self.R, self.R), self.R) )
        N_inv_str   += (str(self.Ninv[len(self.q_i):])[1:-1] + "};")
        Mprimes_str += (str(self.Mprimes[len(self.q_i):])[1:-1] + "};")

        for i in self.q_i:
            one_over_q_str += ("\"" + str(int((Decimal(1)/Decimal(i)*Decimal(2**96)))) + "\"" + ", ")
        one_over_q_str += "0, "
        for i in self.p_i:
            one_over_q_str += ("\"" + str(int((Decimal(1)/Decimal(i)*Decimal(2**96)))) + "\"" + ", ")
        one_over_q_str += "};"

        # q^* = q/q_i
        # q_tilde = (q^*)^-1 mod q_i
        self.q_star    = []
        self.q_tilde   = []
        self.q_tilde_R = []
        for i in self.q_i:
            q_star_i = int(Decimal(self.q) / Decimal(i))
            self.q_star.append(q_star_i)
            self.q_tilde.append(center((modInverse(q_star_i, i)) % i, i))
            self.q_tilde_R.append(center(((modInverse(q_star_i, i)) * self.R) % i, i))
        q_tilde_str   += (str(self.q_tilde)[1:-1] + ", 0, ")
        q_tilde_str_R += (str(self.q_tilde_R)[1:-1] + ", 0, ")
        
        self.p_star    = []
        self.p_tilde   = []
        self.p_tilde_R = []
        for i in self.p_i:
            p_star_i = int(Decimal(self.p) / Decimal(i))
            self.p_star.append(p_star_i)
            self.p_tilde.append(center((modInverse(p_star_i, i)) % i, i))
            self.p_tilde_R.append(center(((modInverse(p_star_i, i)) * self.R) % i, i))
        q_tilde_str   += (str(self.p_tilde)[1:-1] + "};")
        q_tilde_str_R += (str(self.p_tilde_R)[1:-1] + "};")
        
        self.Q_star_q  = []
        self.Q_tilde_q = []
        for i in self.q_i:
            Q_star_qi = int(Decimal(self.Q) / Decimal(i))
            self.Q_star_q.append(Q_star_qi)
            self.Q_tilde_q.append(center((modInverse(Q_star_qi, i)) % i, i))
        
        self.Q_star_p  = []
        self.Q_tilde_p = []
        for i in self.p_i:
            Q_star_pi = int(Decimal(self.Q) / Decimal(i))
            self.Q_star_p.append(Q_star_pi)
            self.Q_tilde_p.append(center((modInverse(Q_star_pi, i)) % i, i))

        self.q_p_mod   = []
        self.q_p_mod_R = []

        for i in self.q_i:
            self.q_p_mod.append(center((self.p) % i, i))
            self.q_p_mod_R.append(center((self.p * self.R) % i, i))
        q_p_mod_str   += (str(self.q_p_mod)[1:-1] + ", 0, ")
        q_p_mod_str_R += (str(self.q_p_mod_R)[1:-1] + ", 0, ")
        for i in self.p_i:
            self.q_p_mod.append(center((self.q) % i, i))
            self.q_p_mod_R.append(center((self.q * self.R) % i, i))
        q_p_mod_str   += (str(self.q_p_mod[len(self.q_i):])[1:-1] + "};")
        q_p_mod_str_R += (str(self.q_p_mod_R[len(self.q_i):])[1:-1] + "};")

        for i, q_i in enumerate(self.q_i):
            star = self.q_star[i]
            q_star_to_p   = []
            q_star_to_p_R = []
            for j, q_j in enumerate(self.p_i):
                q_i_mod_p_j = center((star) % q_j, q_j)
                q_star_to_p.append(q_i_mod_p_j)
                q_i_mod_p_j_R = center((star * self.R) % q_j, q_j)
                q_star_to_p_R.append(q_i_mod_p_j_R)
            q_to_p_str   += ("{" + str(q_star_to_p)[1:-1] + "}, ")
            q_to_p_str_R += ("{" + str(q_star_to_p_R)[1:-1] + "}, ")
        q_to_p_str   += (("{" + "0,"*self.RNS_NUM + "}") + "};")
        q_to_p_str_R += (("{" + "0,"*self.RNS_NUM + "}") + "};")

        for j, q_j in enumerate(self.p_i):
            star = self.p_star[j]
            p_star_to_q   = []
            p_star_to_q_R = []
            for i, q_i in enumerate(self.q_i):
                p_j_mod_q_i = center((star) % q_j, q_j)
                p_star_to_q.append(p_j_mod_q_i)
                p_j_mod_q_i_R = center((star * self.R) % q_j, q_j)
                p_star_to_q_R.append(p_j_mod_q_i_R)
            p_star_to_q.append(0)
            p_star_to_q_R.append(0)
            p_to_q_str   += ("{" + str(p_star_to_q)[1:-1] + "}, ") 
            p_to_q_str_R += ("{" + str(p_star_to_q_R)[1:-1] + "}, ") 
        p_to_q_str   += "};"
        p_to_q_str_R += "};"

        for t in self.t:
            Theta   += "{"
            Omega   += "{"
            Omega_R += "{"
            for i, q_i in enumerate(self.q_i):
                out = (t * modInverse(self.Q_star_q[i], q_i) * self.p)
                out = Decimal(out) / Decimal(q_i)
                omega = int(Decimal(out))
                theta = int((Decimal(out) - Decimal(omega))*Decimal(2**64))
                Theta += ("\"" + str(theta) + "\"" + ", ")
                omega_str   = ""
                omega_R_str = ""
                for j, q_j in enumerate(self.p_i):
                    omega_mod   = center((omega) % q_j, q_j)
                    omega_R_mod = center((omega * self.R) % q_j, q_j)
                    omega_str   += (str(omega_mod) + ", ")
                    omega_R_str += (str(omega_R_mod) + ", ")
                Omega   += ("{" + omega_str + "}, ")
                Omega_R += ("{" + omega_R_str + "}, ")
            Theta   += "},"
            Omega   += "},"
            Omega_R += "},"
        Theta   += "};"
        Omega   += "};"
        Omega_R += "};"

        for t in self.t:
            tQ   += "{"
            tQ_R += "{"
            for j, q_j in enumerate(self.p_i):
                temp   = center((t * modInverse(self.Q_star_p[j], q_j) * self.p_star[j]) % q_j, q_j)
                temp_R = center((t * modInverse(self.Q_star_p[j], q_j) * self.p_star[j] * self.R) % q_j, q_j)
                tQ   += str(temp) + ", "
                tQ_R += str(temp_R) + ", "
            tQ   += "},"
            tQ_R += "},"
        tQ   += "};"
        tQ_R += "};"

        # NTT tables
        self.ntt_forward_table = []
        self.ntt_inverse_table = []
        for mod in self.q_i + self.p_i:
            G = 2
            for i in range(2, mod):
                if is_primitive_root(i, mod):
                    G = i
                    break
            
            TABLE = []
            for i in range(2,int(math.log(self.n, 2))+2):
                tmp = []
                for j in range(0, 2**i, 2):
                    fmt = '{:0' + str(i) + 'b}'
                    bitrev = int(fmt.format(j)[::-1], 2)
                    c = (pow(G, int(((mod-1)/(2**i))*bitrev), mod) * self.R) % mod
                    tmp.append(center(c, mod))
                TABLE.append(tmp[int(len(tmp)/2):])
            self.ntt_forward_table.append(TABLE)

            INV_TABLE = []
            for i in range(2,int(math.log(self.n, 2))+2):
                tmp = []
                for j in range(0, 2**i, 2):
                    fmt = '{:0' + str(i) + 'b}'
                    bitrev = int(fmt.format(j)[::-1], 2)
                    c = (pow(G, ((mod-1)-(bitrev*(mod-1)//(2**i))) % (mod-1), mod) * self.R) % mod
                    tmp.append(center(c, mod))
                INV_TABLE.append(tmp[int(len(tmp)/2):])
            self.ntt_inverse_table.append(INV_TABLE)

            out = []
            for i in (TABLE):
                out = out + i
            out.append(0)
            forward += "{" + str(out)[1:-1] + "},\n"
            
            out = []
            for i in (INV_TABLE):
                out = out + i
            out.append(0)
            inverse += "{" + str(out)[1:-1] + "},\n"
        
        forward += "};"
        inverse += "};"

        self.output_str = """#include "ap_int.h"
#include "ap_fixed.h"

#define LOG_1(n) (((n) >= 2) ? 1 : 0)
#define LOG_2(n) (((n) >= 1<<2) ? (2 + LOG_1((n)>>2)) : LOG_1(n))
#define LOG_4(n) (((n) >= 1<<4) ? (4 + LOG_2((n)>>4)) : LOG_2(n))
#define LOG_8(n) (((n) >= 1<<8) ? (8 + LOG_4((n)>>8)) : LOG_4(n))
#define LOG(n)   (((n) >= 1<<16) ? (16 + LOG_8((n)>>16)) : LOG_8(n))

#define UNROLL_NUM 2      // variable
#define N   """ + str(self.n) + """          // variable
#define PRIME_BIT 32
#define LOG_UNROLL_NUM LOG(UNROLL_NUM)
#define LOG_N LOG(N)
// #define MONTGOMERY

#define R (1<<32)
#define NUM_PLAIN """ + str(len(self.t)) + """
#define RNS_NUM """ + str(self.RNS_NUM) + """
#define LOG_RNS LOG(RNS_NUM)+1

"""
        self.output_str += T_str + "\n"
        self.output_str += delta_str + "\n"
        self.output_str += forward + "\n"
        self.output_str += inverse + "\n"
        self.output_str += "\n"
        self.output_str += mod_str + "\n"
        self.output_str += N_inv_str + "\n"
        self.output_str += Mprimes_str + "\n"
        self.output_str += one_over_q_str + "\n"
        self.output_str += Theta + "\n"
        self.output_str += "\n"
        self.output_str += "#ifdef MONTGOMERY" + "\n"
        self.output_str += q_tilde_str_R + "\n"
        self.output_str += q_p_mod_str_R + "\n"
        self.output_str += q_to_p_str_R + "\n"
        self.output_str += p_to_q_str_R + "\n"
        self.output_str += Omega_R + "\n"
        self.output_str += tQ_R + "\n"
        self.output_str += "#endif" + "\n"
        self.output_str += "\n"
        self.output_str += "#ifndef MONTGOMERY" + "\n"
        self.output_str += q_tilde_str + "\n"
        self.output_str += q_p_mod_str + "\n"
        self.output_str += q_to_p_str + "\n"
        self.output_str += p_to_q_str + "\n"
        self.output_str += Omega + "\n"
        self.output_str += tQ + "\n"
        self.output_str += "#endif" + "\n"
        self.output_str += "\n"
        self.output_str += """void lift(ap_int<2*PRIME_BIT> poly_coeff[2*RNS_NUM][UNROLL_NUM][N/UNROLL_NUM/2], bool inv);
void scale(ap_int<2*PRIME_BIT> poly_coeff[2*RNS_NUM][UNROLL_NUM][N/UNROLL_NUM/2]);"""
    
    def printout(self):
        print(self.output_str)

def Print(v, n):
    ### Print out
    for i in range(2):
        for j in range(n//2):
            sys.stdout.write('%6d ' % v[i*(n//2)+j])
        sys.stdout.write('\n')

### Parameter sets
n = 8192
t = [65537, 114689, 147457, 163841]
q = [31, 31, 31, 31, 31, 31, 31]

# print("Parameters:")
# print(n, t, q)

# vector1 = [(random.randint(-100000, 100000)) for i in range(n)]
# vector2 = [(random.randint(-1000, 1000)) for i in range(n)]
# vector3 = [(random.randint(-1000, 1000)) for i in range(n)]

#context = BFVContext(q, n, t)
# pt1    = context.crt_and_encode(vector1)
# pt2    = context.crt_and_encode(vector2)
# pt2    = context.rns_ntt_pt_addition(pt2)
# pt3    = context.crt_and_encode(vector3)
# pt3    = context.rns_ntt_pt_multiplication(pt3)

#with open("test.txt", "wb") as fp:
#    pickle.dump(context, fp)

# ct     = context.encrypt(pt1)
# context.printBudget(ct)
# ct     = context.HAdd(ct, ct)
# context.printBudget(ct)
# ct     = context.HMul(ct, ct)
# context.printBudget(ct)
# ct     = context.PlainAdd(ct, pt2)
# context.printBudget(ct)
# ct     = context.PlainMul(ct, pt3)
# context.printBudget(ct)
# ct     = context.rotate_row(ct, 1)
# context.printBudget(ct)

# plain_result = [(((i*2)**2)+j)*k for i, j, k in zip(vector1, vector2, vector3)]
# Print(plain_result, n)

# # Test left rotate
# # for idx in range(1, n//2):
# #     print(idx)
# #     ct2     = context.rotate_column(ct, -idx)
# #     context.printBudget(ct2)

# #     pt     = context.decrypt(ct2)
# #     result = context.decode_and_reconstruct(pt)

# #     plain_result = [(((i*2)**2)+j)*k for i, j, k in zip(vector1, vector2, vector3)]
# #     f_result = plain_result[n//2:][idx:] + plain_result[n//2:][:idx] + plain_result[:n//2][idx:] + plain_result[:n//2][:idx]

# #     for i, j in zip(result, f_result):
# #         if i != j:
# #             print(idx, "Error")
# #             break

# # Test right rotate
# # for idx in range(1, n//2):
# #     print(idx)
# #     ct2     = context.rotate_column(ct, idx)
# #     context.printBudget(ct2)

# #     pt     = context.decrypt(ct2)
# #     result = context.decode_and_reconstruct(pt)

# #     plain_result = [(((i*2)**2)+j)*k for i, j, k in zip(vector1, vector2, vector3)]
# #     f_result = plain_result[n//2:][n//2-idx:] + plain_result[n//2:][:n//2-idx] + \
# #                plain_result[:n//2][n//2-idx:] + plain_result[:n//2][:n//2-idx]

# #     for i, j in zip(result, f_result):
# #         if i != j:
# #             print(idx, "Error")
# #             break

# ct  = context.rotate_column(ct, 1)
# context.printBudget(ct)
# pt  = context.decrypt(ct)
# out = context.decode_and_reconstruct(pt)
# Print(out, n)
