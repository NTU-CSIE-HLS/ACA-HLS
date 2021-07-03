#include "ap_int.h"
#include "ap_fixed.h"
#include "rns.h"

#include <iostream>
using namespace std;

extern "C" {

  void K_apply_galois(ap_int<PRIME_BIT> *in, int r, ap_int<PRIME_BIT> *out) {
   #pragma HLS INTERFACE m_axi port=in offset=slave
   #pragma HLS INTERFACE m_axi port=out offset=slave
   #pragma HLS INTERFACE s_axilite port = in bundle = control
   #pragma HLS INTERFACE s_axilite port = r bundle = control
   #pragma HLS INTERFACE s_axilite port = out bundle = control
   #pragma HLS INTERFACE s_axilite port = return bundle = control

     for (ap_uint<LOG_RNS+1> i = 0; i < RNS_NUM; i++) {
     #pragma HLS PIPELINE
         for (ap_uint<LOG_N+1> j = 0; j < N; j++) {
			 ap_int<PRIME_BIT> new_value = in[j + i*N];
			 ap_int<PRIME_BIT> temp = j * r;
			 if (((temp) >> LOG_N) % 2 == 1) // multiply by minus one
				 new_value *= -1;
			 ap_int<PRIME_BIT> new_idx = temp % N;
			 out[new_idx + i*N] = new_value;
         }
     }

     return;
   }
}
