#include "rns.h"
#include "mod.h"
#include <iostream>
using namespace std;


extern "C" {

const int PRAGMA_UNROLL_NUM = UNROLL_NUM;
const int PRAGMA_RNS_NUM_2 = 2*RNS_NUM;
const int PRAGMA_RNS_NUM = RNS_NUM;
const int PRAGMA_LIFT_LATENCY = LIFT_LATENCY;


ap_int<LOG_RNS+1> get_v(ap_int<PRIME_BIT> a[RNS_NUM], const ap_uint<3*PRIME_BIT> b[2*RNS_NUM], bool inv) {
#pragma HLS PIPELINE II=PRAGMA_RNS_NUM
    ap_int<3*PRIME_BIT+LOG_RNS+1> acc = 0;
    for (int j = 0; j < RNS_NUM; j++) {
        ap_uint<LOG_RNS+1> idx = inv ? j+RNS_NUM : j;
        ap_int<LOG_RNS+1+3*PRIME_BIT> temp = a[j] * (ap_int<LOG_RNS+1+3*PRIME_BIT>)b[idx];
        acc += temp;
    }
    ap_int<LOG_RNS+1> ret;
    ret = (acc(3*PRIME_BIT-1, 3*PRIME_BIT-1)) ? acc(3*PRIME_BIT+LOG_RNS, 3*PRIME_BIT) + 1 :
                                                acc(3*PRIME_BIT+LOG_RNS, 3*PRIME_BIT);
    return ret;
}

ap_int<PRIME_BIT+1> calc(ap_int<LOG_RNS+1> v,
                         ap_int<PRIME_BIT> temp[RNS_NUM],
                         ap_uint<LOG_N> tmp,
                         ap_uint<PRIME_BIT> mod,
                         ap_int<PRIME_BIT> mprime,
                         ap_int<PRIME_BIT> q_p,
                         bool inv, unsigned int idx)
{
#pragma HLS PIPELINE II=PRAGMA_LIFT_LATENCY
#pragma HLS ALLOCATION instances=mul_mod limit=2 function

    ap_uint<LOG_N> k;
    ap_int<PRIME_BIT+1> acc = 0;
    for (k = 0; k < RNS_NUM; k++) {
        ap_int<PRIME_BIT> multiplicand = inv ? p_to_q[k][tmp] : q_to_p[k][tmp];
        ap_int<PRIME_BIT> temp22 = modular_multiplication(temp[k], multiplicand, mod, idx);
        acc += temp22;
        if (acc > (mod>>1))
            acc -= mod;
        if (acc < -(mod>>1))
            acc += mod;
    }

    ap_int<PRIME_BIT> temp3 = modular_multiplication(v, q_p, mod, idx);
    acc = acc - temp3;
    if (acc > (mod>>1))
        acc -= mod;
    if (acc < -(mod>>1))
        acc += mod;

    return acc;
}

void lift(ap_uint<2*PRIME_BIT> poly_coeff[RNS_NUM][UNROLL_NUM][N/UNROLL_NUM/2],
          ap_uint<2*PRIME_BIT> out[RNS_NUM][UNROLL_NUM][N/UNROLL_NUM/2],
          bool inv)
{
#pragma HLS INTERFACE bram port=poly_coeff
#pragma HLS RESOURCE variable=poly_coeff core=RAM_T2P_BRAM
#pragma HLS ARRAY_PARTITION variable=poly_coeff complete dim=1
#pragma HLS ARRAY_PARTITION variable=poly_coeff block factor=PRAGMA_UNROLL_NUM dim=2
#pragma HLS ALLOCATION instances=modular_multiplication limit=2 function

    ap_uint<LOG_N> i, j, l;
    main: for (i = 0; i < N/UNROLL_NUM/2; i++) {
#pragma HLS PIPELINE II=PRAGMA_LIFT_LATENCY
        ap_int<PRIME_BIT> temp1[UNROLL_NUM][RNS_NUM], temp2[UNROLL_NUM][RNS_NUM];
#pragma HLS ARRAY_PARTITION variable=temp1 complete dim=1
#pragma HLS ARRAY_PARTITION variable=temp1 complete dim=2
#pragma HLS ARRAY_PARTITION variable=temp2 complete dim=1
#pragma HLS ARRAY_PARTITION variable=temp2 complete dim=2

        calc_y: for (j = 0; j < RNS_NUM; j++) {
            parallel_unroll: for (l = 0; l < UNROLL_NUM; l++) {
            //#pragma HLS UNROLL
                ap_uint<LOG_N> tmp = j+RNS_NUM;
                ap_uint<LOG_N> idx = (inv ? tmp : j);
                ap_uint<2*PRIME_BIT> tempp = poly_coeff[j][l][i];
                unsigned int idx2 = inv ? tmp : j;
                temp1[l][j] = modular_multiplication(tempp(PRIME_BIT-1, 0), q_tilde[idx], MOD[idx], idx2);
                temp2[l][j] = modular_multiplication(tempp(2*PRIME_BIT-1, PRIME_BIT), q_tilde[idx], MOD[idx], idx2);
            }
        }

        ap_int<LOG_RNS+1> v1[UNROLL_NUM], v2[UNROLL_NUM];

        parallel_unroll2: for (l = 0; l < UNROLL_NUM; l++) {
        #pragma HLS UNROLL
            v1[l] = get_v(temp1[l], one_over_q, inv);
            v2[l] = get_v(temp2[l], one_over_q, inv);
        }

        calculate_unroll1: for (j = RNS_NUM; j < 2*RNS_NUM; j++) {
        #pragma HLS UNROLL
            parallel_unroll3: for (l = 0; l < UNROLL_NUM; l++) {
            //#pragma HLS UNROLL
                ap_uint<LOG_N> tmp = j-RNS_NUM;
                ap_uint<LOG_N> idx1 = (inv ? tmp : j);

                ap_int<PRIME_BIT+1> acc1 = calc(v1[l], temp1[l], tmp, MOD[idx1], Mprime[idx1], q_p[idx1], inv, idx1);
                ap_int<PRIME_BIT+1> acc2 = calc(v2[l], temp2[l], tmp, MOD[idx1], Mprime[idx1], q_p[idx1], inv, idx1);

                out[tmp][l][i] = (acc2(PRIME_BIT-1,0), acc1(PRIME_BIT-1,0));
            }
        }
    }
}


void K_lift(ap_int<PRIME_BIT> *in,
		    ap_int<PRIME_BIT> *out,
		    int which)
{
#pragma HLS ALLOCATION instances=lift limit=1 function
#pragma HLS INTERFACE m_axi port=in offset=slave
#pragma HLS INTERFACE m_axi port=out offset=slave
#pragma HLS INTERFACE s_axilite port = in bundle = control
#pragma HLS INTERFACE s_axilite port = out bundle = control
#pragma HLS INTERFACE s_axilite port = which bundle = control
#pragma HLS INTERFACE s_axilite port = return bundle = control

	ap_uint<2*PRIME_BIT> x[RNS_NUM][UNROLL_NUM][N/UNROLL_NUM/2];
	ap_uint<2*PRIME_BIT> y[RNS_NUM][UNROLL_NUM][N/UNROLL_NUM/2];
#pragma HLS ARRAY_PARTITION variable=x complete dim=1
#pragma HLS ARRAY_PARTITION variable=x complete dim=2
#pragma HLS ARRAY_PARTITION variable=y complete dim=1
#pragma HLS ARRAY_PARTITION variable=y complete dim=2

	for (ap_uint<LOG_RNS+1> r = 0; r < RNS_NUM; r++) {
#pragma HLS PIPELINE
		for (ap_uint<LOG_N+1> i = 0; i < N/2; i++) {
			x[r][i/(N/UNROLL_NUM/2)][i%(N/UNROLL_NUM/2)] = (in[2*i+1+r*N], in[2*i+r*N]);
		}
	}
	lift(x, y, which);
	for (ap_uint<LOG_RNS+1> r = 0; r < RNS_NUM; r++) {
#pragma HLS PIPELINE
		for (ap_uint<LOG_N+1> i = 0; i < N/2; i++) {
			(out[2*i+1+r*N], out[2*i+r*N]) = x[r][i/(N/UNROLL_NUM/2)][i%(N/UNROLL_NUM/2)];
		}
	}

	return;
}



}
