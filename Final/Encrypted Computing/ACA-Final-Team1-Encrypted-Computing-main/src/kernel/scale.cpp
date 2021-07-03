#include "rns.h"
#include "mod.h"
#include <iostream>
using namespace std;

extern "C" {

const int PRAGMA_UNROLL_NUM = UNROLL_NUM;
const int PRAGMA_RNS_NUM_2_MINUS_1 = 2*RNS_NUM-1;
const int PRAGMA_RNS_NUM_2 = 2*RNS_NUM;
const int PRAGMA_RNS_NUM = RNS_NUM;
const int PRAGMA_SCALE_LATENCY = SCALE_LATENCY;

ap_int<PRIME_BIT+1> center(ap_int<PRIME_BIT+1> a, ap_uint<PRIME_BIT> mod)
{
	ap_int<PRIME_BIT + 1> t = a;
    if (t > mod/2)
        t -= mod;
    if (t < -mod/2)
        t += mod;
    return t;
}

void scale(ap_uint<2*PRIME_BIT> poly_coeff1[RNS_NUM][UNROLL_NUM][N/UNROLL_NUM/2],
           ap_uint<2*PRIME_BIT> poly_coeff2[RNS_NUM][UNROLL_NUM][N/UNROLL_NUM/2],
           ap_uint<2*PRIME_BIT>         out[RNS_NUM][UNROLL_NUM][N/UNROLL_NUM/2],
           ap_uint<LOG_RNS> which) {
#pragma HLS INTERFACE bram port=poly_coeff1
#pragma HLS RESOURCE variable=poly_coeff1 core=RAM_T2P_BRAM
#pragma HLS ARRAY_PARTITION variable=poly_coeff1 complete dim=1
#pragma HLS ARRAY_PARTITION variable=poly_coeff1 block factor=PRAGMA_UNROLL_NUM dim=2
#pragma HLS INTERFACE bram port=poly_coeff2
#pragma HLS RESOURCE variable=poly_coeff2 core=RAM_T2P_BRAM
#pragma HLS ARRAY_PARTITION variable=poly_coeff2 complete dim=1
#pragma HLS ARRAY_PARTITION variable=poly_coeff2 block factor=PRAGMA_UNROLL_NUM dim=2
#pragma HLS INTERFACE bram port=out
#pragma HLS RESOURCE variable=out core=RAM_T2P_BRAM
#pragma HLS ARRAY_PARTITION variable=out complete dim=1
#pragma HLS ARRAY_PARTITION variable=out block factor=PRAGMA_UNROLL_NUM dim=2

    ap_uint<LOG_N> i, j, k, l;
    main: for (i = 0; i < N/UNROLL_NUM/2; i++) {
#pragma HLS PIPELINE II=PRAGMA_SCALE_LATENCY
        for (j = RNS_NUM; j < 2*RNS_NUM; j++) {

            unroll_calculate: for (l = 0; l < UNROLL_NUM; l++) {
            //#pragma HLS UNROLL, put it in the innerest loop
                ap_uint<LOG_N> tmp = j-RNS_NUM;

                ap_int<3*PRIME_BIT+1> acc11 = 0;
                ap_int<3*PRIME_BIT+1> acc22 = 0;
                ap_int<PRIME_BIT+1> acc1 = 0;
                ap_int<PRIME_BIT+1> acc2 = 0;

                unroll_k: for (k = 0; k < RNS_NUM-1; k++) {
                    // get coeff
                    ap_uint<2*PRIME_BIT> tempp = poly_coeff1[k][l][i];
                    ap_int<PRIME_BIT> temp1 = (ap_int<PRIME_BIT>)tempp(PRIME_BIT-1, 0);
                    ap_int<PRIME_BIT> temp2 = (ap_int<PRIME_BIT>)tempp(2*PRIME_BIT-1, PRIME_BIT);

                    // Theta
                    ap_int<3*PRIME_BIT> temp11 = temp1*Theta[which][k];
                    acc11 += temp11;
                    ap_int<PRIME_BIT+1> upper1 = acc11(3*PRIME_BIT, 2*PRIME_BIT);
                    upper1 = center(upper1, MOD[j]);
                    acc11 = (upper1, acc11(2*PRIME_BIT-1, 0));

                    ap_int<3*PRIME_BIT> temp22 = temp2*Theta[which][k];
                    acc22 += temp22;
                    ap_int<PRIME_BIT+1> upper2 = acc22(3*PRIME_BIT, 2*PRIME_BIT);
                    upper2 = center(upper2, MOD[j]);
                    acc22 = (upper2, acc22(2*PRIME_BIT-1, 0));

                    // Omega
                    ap_int<PRIME_BIT> multiplicand = Omega[which][k][tmp];
                    ap_int<PRIME_BIT> temp1111 = modular_multiplication(temp1, multiplicand, MOD[j], j);
                    acc1 += temp1111;
                    acc1 = center(acc1, MOD[j]);
                    ap_int<PRIME_BIT> temp2222 = modular_multiplication(temp2, multiplicand, MOD[j], j);
                    acc2 += temp2222;
                    acc2 = center(acc2, MOD[j]);
                }
                ap_uint<2*PRIME_BIT> temppp = poly_coeff2[j-RNS_NUM][l][i];
                ap_int<PRIME_BIT> tta = temppp(PRIME_BIT-1, 0);
                ap_int<PRIME_BIT> ttb = temppp(2*PRIME_BIT-1, PRIME_BIT);
                ap_int<PRIME_BIT> out1 = modular_multiplication(tta, tQ[which][tmp], MOD[j], j);
                ap_int<PRIME_BIT> out2 = modular_multiplication(ttb, tQ[which][tmp], MOD[j], j);
                acc1 += out1;
                acc1 = center(acc1, MOD[j]);
                acc2 += out2;
                acc2 = center(acc2, MOD[j]);
                ap_int<PRIME_BIT> ret1;
                ret1 = (acc11(2*PRIME_BIT-1, 2*PRIME_BIT-1)) ? acc11(3*PRIME_BIT-1, 2*PRIME_BIT) + 1 :
                                                               acc11(3*PRIME_BIT-1, 2*PRIME_BIT);
                ap_int<PRIME_BIT> ret2;
                ret2 = (acc22(2*PRIME_BIT-1, 2*PRIME_BIT-1)) ? acc22(3*PRIME_BIT-1, 2*PRIME_BIT) + 1 :
                                                               acc22(3*PRIME_BIT-1, 2*PRIME_BIT);
                acc1 += ret1;
                acc1 = center(acc1, MOD[j]);
                acc2 += ret2;
                acc2 = center(acc2, MOD[j]);
                out[j-RNS_NUM][l][i] = (acc2(PRIME_BIT-1,0), acc1(PRIME_BIT-1,0));
            }

        }
    }
}

void K_scale(ap_int<PRIME_BIT> *in1,
		     ap_int<PRIME_BIT> *in2,
			 ap_int<PRIME_BIT> *out,
		     int which)
{
#pragma HLS ALLOCATION instances=scale limit=1 function
#pragma HLS INTERFACE m_axi port=in1 offset=slave
#pragma HLS INTERFACE m_axi port=in2 offset=slave
#pragma HLS INTERFACE m_axi port=out offset=slave
#pragma HLS INTERFACE s_axilite port = in1 bundle = control
#pragma HLS INTERFACE s_axilite port = in2 bundle = control
#pragma HLS INTERFACE s_axilite port = out bundle = control
#pragma HLS INTERFACE s_axilite port = which bundle = control
#pragma HLS INTERFACE s_axilite port = return bundle = control

	ap_uint<2*PRIME_BIT> poly_coeff1[RNS_NUM][UNROLL_NUM][N/UNROLL_NUM/2];
	ap_uint<2*PRIME_BIT> poly_coeff2[RNS_NUM][UNROLL_NUM][N/UNROLL_NUM/2];
#pragma HLS ARRAY_PARTITION variable=poly_coeff1 complete dim=1
#pragma HLS ARRAY_PARTITION variable=poly_coeff1 complete dim=2
#pragma HLS ARRAY_PARTITION variable=poly_coeff2 complete dim=1
#pragma HLS ARRAY_PARTITION variable=poly_coeff2 complete dim=2

	for (ap_uint<LOG_RNS+1> r = 0; r < RNS_NUM; r++) {
#pragma HLS PIPELINE
		for (ap_uint<LOG_N+1> i = 0; i < N/2; i++) {
			poly_coeff1[r][i/(N/UNROLL_NUM/2)][i%(N/UNROLL_NUM/2)] = (in1[2*i+1+r*N], in1[2*i+r*N]);
			poly_coeff2[r][i/(N/UNROLL_NUM/2)][i%(N/UNROLL_NUM/2)] = (in2[2*i+1+r*N], in2[2*i+r*N]);
		}
	}
	scale(poly_coeff1, poly_coeff2, poly_coeff2, which);
	for (ap_uint<LOG_RNS+1> r = 0; r < RNS_NUM; r++) {
#pragma HLS PIPELINE
		for (ap_uint<LOG_N+1> i = 0; i < N/2; i++) {
			(out[2*i+1+r*N], out[2*i+r*N]) = poly_coeff2[r][i/(N/UNROLL_NUM/2)][i%(N/UNROLL_NUM/2)];
		}
	}

	return;
}

}
