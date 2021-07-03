#include "rns.h"
#include "ap_int.h"
#include "ap_fixed.h"
#include <iostream>
using namespace std;
extern "C" {
  ap_int<PRIME_BIT+1> center_(ap_int<PRIME_BIT+1> a, ap_uint<PRIME_BIT> mod)
  {
   ap_int<PRIME_BIT + 1> t = a;
   if (t > mod/2)
    t -= mod;
  if (t < -mod/2)
    t += mod;
  return t;
}
void K_add(ap_int<PRIME_BIT> *poly1,
 ap_int<PRIME_BIT> *poly2,
 ap_int<PRIME_BIT> *out,
 int inv)
{
  #pragma HLS INTERFACE m_axi port=poly1 offset=slave
  #pragma HLS INTERFACE m_axi port=poly2 offset=slave
  #pragma HLS INTERFACE m_axi port=out offset=slave
  #pragma HLS INTERFACE s_axilite port = poly1 bundle = control
  #pragma HLS INTERFACE s_axilite port = poly2 bundle = control
  #pragma HLS INTERFACE s_axilite port = out bundle = control
  #pragma HLS INTERFACE s_axilite port = inv bundle = control
  #pragma HLS INTERFACE s_axilite port = return bundle = control

  for (int i = 0; i < N*RNS_NUM; i++) {
    #pragma HLS PIPELINE
    ap_int<PRIME_BIT+1> temp;
    temp = poly1[i] + poly2[i];
    ap_int<LOG_RNS+1> idx = (i>>LOG_N) + RNS_NUM;
    ap_int<LOG_RNS+1> idx2 = (i>>LOG_N);
    ap_int<LOG_RNS+1> idx3 = (inv==1) ? idx : idx2;

    ap_int<PRIME_BIT+1> temp2;
    temp2 = center_(temp, MOD[idx3]);
    out[i] = temp2(PRIME_BIT-1, 0);
  }
  return;
}
}
