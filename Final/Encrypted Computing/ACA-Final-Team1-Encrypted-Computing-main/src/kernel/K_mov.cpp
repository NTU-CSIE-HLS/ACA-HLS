#include "rns.h"
#include "ap_int.h"
#include "ap_fixed.h"


extern "C" {

void K_mov(ap_int<PRIME_BIT> *in, ap_int<PRIME_BIT> *out){
#pragma HLS INTERFACE m_axi port=in offset=slave
#pragma HLS INTERFACE m_axi port=out offset=slave
#pragma HLS INTERFACE s_axilite port = in bundle = control
#pragma HLS INTERFACE s_axilite port = out bundle = control
#pragma HLS INTERFACE s_axilite port = return bundle = control

  for (ap_uint<LOG_RNS+LOG_N+1> i = 0; i < RNS_NUM*N; i++) {
  #pragma HLS PIPELINE
    out[i] = in[i];
  }

  return;
}
}
