//#include "ntt.h"
#include "rns.h"
#include <iostream>

using namespace std;

extern "C" {
const int PRAGMA_UNROLL_NUM_NTT = UNROLL_NUM_NTT;
const int PRAGMA_RNS_NUM = RNS_NUM;

void add_sub(ap_int<PRIME_BIT> *a, ap_int<PRIME_BIT> *b, ap_uint<PRIME_BIT> mod) {
    ap_int<PRIME_BIT+1> ta, tb;
    ta = *a + *b;
    tb = *a - *b;
    if (ta > mod/2)
        ta -= mod;
    if (ta < -mod/2)
        ta += mod;
    if (tb > mod/2)
        tb -= mod;
    if (tb < -mod/2)
        tb += mod;
    *a = ta;
    *b = tb;
}

ap_int<PRIME_BIT> mul_mod(ap_int<PRIME_BIT> a,
                          ap_int<PRIME_BIT> b,
                          ap_uint<PRIME_BIT> mod,
                          ap_int<PRIME_BIT> mprime)
{
    ap_int<2*PRIME_BIT> temp = ((a) * (b));
    ap_int<PRIME_BIT> lower = (ap_int<PRIME_BIT>)temp(PRIME_BIT-1, 0) * (ap_int<PRIME_BIT>)mprime;
    ap_int<2*PRIME_BIT> acc = lower * mod;
    acc = acc + temp;
    ap_int<PRIME_BIT> ret = acc(2*PRIME_BIT-1, PRIME_BIT);
    if (ret > mod/2)
        ret -= mod;
    if (ret < -mod/2)
        ret += mod;
    return ret;
}

void butterfly(ap_uint<2*PRIME_BIT> ta,
               ap_int<PRIME_BIT> c,
               bool inv,
               ap_int<PRIME_BIT> *ttta,
               ap_int<PRIME_BIT> *tttb,
               ap_uint<PRIME_BIT> mod,
               ap_int<PRIME_BIT> mprime)
{
#pragma HLS ALLOCATION instances=mul_mod limit=1 function
#pragma HLS ALLOCATION instances=add_sub limit=2 function

    ap_int<PRIME_BIT> tta, ttb;
    tta = ta(PRIME_BIT-1, 0);
    ttb = ta(2*PRIME_BIT-1, PRIME_BIT);
    if (inv) {
        add_sub(&tta, &ttb, mod);
        ttb = mul_mod(ttb, c, mod, mprime);
    } else {
        ttb = mul_mod(ttb, c, mod, mprime);
        add_sub(&tta, &ttb, mod);
    }
    *ttta = tta;
    *tttb = ttb;
}

void NTT(ap_uint<2*PRIME_BIT> poly_coeff[UNROLL_NUM_NTT][N/UNROLL_NUM_NTT/2],
         bool inv,
         ap_uint<PRIME_BIT> mod,
         ap_int<PRIME_BIT>  mprime,
		 const int f_table[N],
		 const int i_table[N])
{
#pragma HLS RESOURCE variable=poly_coeff core=RAM_T2P_BRAM
#pragma HLS ARRAY_PARTITION variable=poly_coeff block factor=PRAGMA_UNROLL_NUM_NTT dim=1
#pragma HLS INTERFACE bram port=poly_coeff
#pragma HLS INTERFACE s_axilite port=inv
#pragma HLS ALLOCATION instances=butterfly limit=PRAGMA_UNROLL_NUM_NTT function

    ap_uint<5> i, l;
    ap_uint<LOG_N> temp_idx;
    ap_uint<LOG_N> half_num_items = N/UNROLL_NUM_NTT/4;
    ap_uint<LOG_N> gap = inv ? 1 : N/4;

    ap_int<PRIME_BIT> twiddle_factor_buffer[UNROLL_NUM_NTT];
    ap_int<PRIME_BIT> buffer0[UNROLL_NUM_NTT];
    ap_int<PRIME_BIT> buffer1[UNROLL_NUM_NTT];
    ap_int<PRIME_BIT> buffer2[UNROLL_NUM_NTT];
    ap_int<PRIME_BIT> buffer3[UNROLL_NUM_NTT];
    ap_int<PRIME_BIT> buffer4[UNROLL_NUM_NTT];

    stage:for(i = 0; i < LOG_N; i++) {

    	group:for(ap_int<LOG_N+1> iii = 0; iii < 2*half_num_items; iii++) {
#pragma HLS DEPENDENCE variable=poly_coeff inter false
#pragma HLS DEPENDENCE variable=poly_coeff intra WAW false
#pragma HLS DEPENDENCE variable=poly_coeff intra WAR true
#pragma HLS DEPENDENCE variable=poly_coeff intra RAW false
#pragma HLS PIPELINE II=1

            if ((i == LOG_N-1 && inv == 0) || (i == 0 && inv == 1)) {
                unroll_1: for(l = 0; l < UNROLL_NUM_NTT; l++) {
                #pragma HLS UNROLL
                    ap_int<PRIME_BIT> twiddle_factor = inv ? i_table[(1<<(LOG_N-1))-1+iii+(1<<(LOG_N-1-LOG_UNROLL_NUM_NTT))*l]:
                            f_table[(1<<(LOG_N-1))-1+iii+(1<<(LOG_N-1-LOG_UNROLL_NUM_NTT))*l];
                    ap_int<PRIME_BIT> t1, t2;
                    butterfly(poly_coeff[l][iii], twiddle_factor, inv, &t1, &t2, mod, mprime);
                    poly_coeff[l][iii] = (t2, t1);
                }
            }
#if UNROLL_NUM_NTT >= 2
            else if ((i == 0 && inv == 0) || (i == LOG_N-1 && inv == 1)) {
                unroll_2: for(l = 0; l < UNROLL_NUM_NTT/2; l++) {
                #pragma HLS UNROLL
                    ap_uint<LOG_N> idx_gap = (1<<(LOG_UNROLL_NUM_NTT-1));
                    ap_uint<LOG_N> tmp0 = l / idx_gap;
                    ap_uint<LOG_N> tmp1 = l % idx_gap;
                    ap_uint<LOG_N> a = (1<<idx_gap)*tmp0 + tmp1;
                    ap_uint<LOG_N> b = (1<<idx_gap)*tmp0 + tmp1 + idx_gap;
                    ap_int<PRIME_BIT> twiddle_factor = inv ? i_table[(1<<0)-1+tmp0] : f_table[(1<<0)-1+tmp0];
                    ap_uint<LOG_N> current_idx = (iii>>1)+half_num_items*(iii&1);
                    ap_int<PRIME_BIT> b0, b1, b2, b3;
                    ap_int<PRIME_BIT> t0, t1, t2, t3;
                    (b1, b0) = poly_coeff[a][current_idx];
                    (b3, b2) = poly_coeff[b][current_idx];
                    ap_int<PRIME_BIT> first_upper  = inv ? b2 : b1;
                    ap_int<PRIME_BIT> second_lower = inv ? b1 : b2;
                    butterfly((first_upper, b0),  twiddle_factor, inv, &t0, &t1, mod, mprime);
                    butterfly((b3, second_lower), twiddle_factor, inv, &t2, &t3, mod, mprime);
                    ap_int<PRIME_BIT> first_upper_2  = inv ? t1 : t2;
                    ap_int<PRIME_BIT> second_lower_2 = inv ? t2 : t1;
                    poly_coeff[a][current_idx] = (first_upper_2, t0);
                    poly_coeff[b][current_idx] = (t3, second_lower_2);

                }
            }
#endif
#if UNROLL_NUM_NTT >= 4
            else if ((i == 1 && inv == 0) || (i == LOG_N-2 && inv == 1)) {
                unroll_4: for(l = 0; l < UNROLL_NUM_NTT/2; l++) {
                #pragma HLS UNROLL
                    ap_uint<LOG_N> idx_gap = (1<<(LOG_UNROLL_NUM_NTT-2));
                    ap_uint<LOG_N> tmp0 = l / idx_gap;
                    ap_uint<LOG_N> tmp1 = l % idx_gap;
                    ap_uint<LOG_N> a = (1<<idx_gap)*tmp0 + tmp1;
                    ap_uint<LOG_N> b = (1<<idx_gap)*tmp0 + tmp1 + idx_gap;
                    ap_int<PRIME_BIT> twiddle_factor = inv ? i_table[(1<<1)-1+tmp0] : f_table[(1<<1)-1+tmp0];
                    ap_uint<LOG_N> current_idx = (iii>>1)+half_num_items*(iii&1);
                    ap_int<PRIME_BIT> b0, b1, b2, b3;
                    ap_int<PRIME_BIT> t0, t1, t2, t3;
                    (b1, b0) = poly_coeff[a][current_idx];
                    (b3, b2) = poly_coeff[b][current_idx];
                    ap_int<PRIME_BIT> first_upper  = inv ? b2 : b1;
                    ap_int<PRIME_BIT> second_lower = inv ? b1 : b2;
                    butterfly((first_upper, b0),  twiddle_factor, inv, &t0, &t1, mod, mprime);
                    butterfly((b3, second_lower), twiddle_factor, inv, &t2, &t3, mod, mprime);
                    ap_int<PRIME_BIT> first_upper_2  = inv ? t1 : t2;
                    ap_int<PRIME_BIT> second_lower_2 = inv ? t2 : t1;
                    poly_coeff[a][current_idx] = (first_upper_2, t0);
                    poly_coeff[b][current_idx] = (t3, second_lower_2);
                }
            }
#endif
#if UNROLL_NUM_NTT >= 8
            else if ((i == 2 && inv == 0) || (i == LOG_N-3 && inv == 1)) {
                unroll_8: for(l = 0; l < UNROLL_NUM_NTT/2; l++) {
                #pragma HLS UNROLL
                    ap_uint<LOG_N> idx_gap = (1<<(LOG_UNROLL_NUM_NTT-3));
                    ap_uint<LOG_N> tmp0 = l / idx_gap;
                    ap_uint<LOG_N> tmp1 = l % idx_gap;
                    ap_uint<LOG_N> a = (1<<idx_gap)*tmp0 + tmp1;
                    ap_uint<LOG_N> b = (1<<idx_gap)*tmp0 + tmp1 + idx_gap;
                    ap_int<PRIME_BIT> twiddle_factor = inv ? i_table[(1<<2)-1+tmp0] : f_table[(1<<2)-1+tmp0];
                    ap_uint<LOG_N> current_idx = (iii>>1)+half_num_items*(iii&1);
                    ap_int<PRIME_BIT> b0, b1, b2, b3;
                    ap_int<PRIME_BIT> t0, t1, t2, t3;
                    (b1, b0) = poly_coeff[a][current_idx];
                    (b3, b2) = poly_coeff[b][current_idx];
                    ap_int<PRIME_BIT> first_upper  = inv ? b2 : b1;
                    ap_int<PRIME_BIT> second_lower = inv ? b1 : b2;
                    butterfly((first_upper, b0),  twiddle_factor, inv, &t0, &t1, mod, mprime);
                    butterfly((b3, second_lower), twiddle_factor, inv, &t2, &t3, mod, mprime);
                    ap_int<PRIME_BIT> first_upper_2  = inv ? t1 : t2;
                    ap_int<PRIME_BIT> second_lower_2 = inv ? t2 : t1;
                    poly_coeff[a][current_idx] = (first_upper_2, t0);
                    poly_coeff[b][current_idx] = (t3, second_lower_2);
                }
            }
#endif
            else {
                ap_uint<LOG_N> k = (iii>>1) % gap;
                ap_uint<LOG_N> j = (iii>>1) / gap;
                ap_uint<LOG_N> idx0 = j*(gap<<1) + k;
                ap_uint<LOG_N> idx1 = j*(gap<<1) + k + gap;

                unroll_normal: for (l = 0; l < UNROLL_NUM_NTT; l++) {
                #pragma HLS UNROLL
                    ap_int<PRIME_BIT> twiddle_factor = inv ? i_table[(1<<(LOG_N-1-i))-1+j+l*(1<<(LOG_N-LOG_UNROLL_NUM_NTT-1-i))] :
                                                f_table[(1<<(i))-1+j+l*(1<<(i-LOG_UNROLL_NUM_NTT))];
                    if (inv == 0) {
                        ap_uint<2*PRIME_BIT> t = (iii % 2 == 0) ? poly_coeff[l][idx0] : poly_coeff[l][idx1];
                        ap_int<PRIME_BIT> t1, t2;
                        butterfly(t, twiddle_factor, inv, &t1, &t2, mod, mprime);
                        buffer0[l] = (iii % 2 == 0) ? (t1) : buffer0[l];
                        buffer1[l] = (iii % 2 == 0) ? (t2) : buffer1[l];
                        buffer2[l] = (iii % 2 == 0) ? buffer2[l] : t1;
                        buffer3[l] = (iii % 2 == 0) ? buffer3[l] : t2;
                        ap_uint<LOG_N> store_idx = ((iii == 0) || (iii % 2 == 1)) ? idx0 : temp_idx;
                        poly_coeff[l][store_idx] = (iii == 0) ? t :
                                ((iii % 2 == 0) ? (buffer3[l], buffer4[l]) : (buffer2[l], buffer0[l]));
                        buffer4[l] = buffer1[l];
                    } else {
                        ap_int<PRIME_BIT> b1, b2;
                        (b2, b1) = (iii % 2 == 0) ? poly_coeff[l][idx0] : poly_coeff[l][idx1];
                        ap_int<PRIME_BIT> t1, t2;
                        ap_int<PRIME_BIT> factor = (iii % 2 == 0) ? twiddle_factor_buffer[l] : twiddle_factor;
                        ap_uint<2*PRIME_BIT> tt = (iii % 2 == 0) ? (buffer3[l], buffer4[l]) : (b1, buffer0[l]);
                        butterfly(tt, factor, inv, &t1, &t2, mod, mprime);
                        ap_uint<LOG_N> store_idx = ((iii == 0)||(iii % 2 == 1)) ? idx0 : temp_idx;
                        poly_coeff[l][store_idx] = iii == 0 ? (b2, b1) : (t2, t1);
                        buffer0[l] = (iii % 2 == 0) ? (b1) : buffer0[l];
                        buffer1[l] = (iii % 2 == 0) ? (b2) : buffer1[l];
                        buffer2[l] = (iii % 2 == 0) ? buffer2[l] : b1;
                        buffer3[l] = (iii % 2 == 0) ? buffer3[l] : b2;
                        buffer4[l] = (iii % 2 == 0) ? buffer4[l] : buffer1[l];
                        twiddle_factor_buffer[l] = twiddle_factor;
                    }
                }
                temp_idx = idx1;
            }

        }
        unroll_remaining: for(l = 0; l < UNROLL_NUM_NTT; l++) {
        #pragma HLS UNROLL
            if ((i >= LOG_UNROLL_NUM_NTT && i < LOG_N - 1 && inv == 0)) {
                poly_coeff[l][temp_idx] = (buffer3[l], buffer4[l]);
            } else if ((i < LOG_N - LOG_UNROLL_NUM_NTT && i > 0 && inv == 1)) {
                ap_uint<2*PRIME_BIT> tt = (buffer3[l], buffer4[l]);
                ap_int<PRIME_BIT> t1, t2;
                butterfly(tt, twiddle_factor_buffer[l], inv, &t1, &t2, mod, mprime);
                poly_coeff[l][temp_idx] = (t2, t1);
            }
        }
        gap = inv ? ((i == 0) ? gap<<0 : gap<<1) : gap>>1;
    }
}

void K_ntt(ap_int<PRIME_BIT> *in, ap_int<PRIME_BIT> *out, int mode, int which_poly)
{
#pragma HLS ALLOCATION instances=NTT limit=1 function
#pragma HLS INTERFACE m_axi port=in offset=slave
#pragma HLS INTERFACE m_axi port=out offset=slave
#pragma HLS INTERFACE s_axilite port = in bundle = control
#pragma HLS INTERFACE s_axilite port = out bundle = control
#pragma HLS INTERFACE s_axilite port = mode bundle = control
#pragma HLS INTERFACE s_axilite port = which_poly bundle = control
#pragma HLS INTERFACE s_axilite port = return bundle = control
    /**
     * mode 0: forward 7
     *      1: inverse 7
     *      2: forward 8
     *      3: inverse 8
     *      4: decompose one poly to multiple NTT'd poly
     *
     * which poly
     **/

    ap_uint<2*PRIME_BIT> poly_coeff[RNS_NUM][UNROLL_NUM_NTT][N/UNROLL_NUM_NTT/2];
#pragma HLS ARRAY_PARTITION variable=poly_coeff complete dim=1
#pragma HLS ARRAY_PARTITION variable=poly_coeff complete dim=2

    if (mode == 0) {
    	for (ap_uint<LOG_RNS+1> r = 0; r < (RNS_NUM-1); r++) {
#pragma HLS PIPELINE
    		for (ap_uint<LOG_N+1> i = 0; i < N/2; i++) {
    			poly_coeff[r][i/(N/UNROLL_NUM_NTT/2)][i%(N/UNROLL_NUM_NTT/2)] = (in[i+N/2+r*N], in[i+r*N]);
    		}
    	}
        for (ap_uint<LOG_RNS+1> i = 0; i < RNS_NUM-1; i++) {
        	NTT(poly_coeff[i], false, MOD[i], Mprime[i], forward_table[i], inv_table[i]);
        }
        for (ap_uint<LOG_RNS+1> r = 0; r < (RNS_NUM-1); r++) {
        #pragma HLS PIPELINE
			for (ap_uint<LOG_N+1> i = 0; i < N/2; i++) {
				out[2*i+r*N]   = poly_coeff[r][i/(N/UNROLL_NUM_NTT/2)][i%(N/UNROLL_NUM_NTT/2)](PRIME_BIT-1, 0);
				out[2*i+r*N+1] = poly_coeff[r][i/(N/UNROLL_NUM_NTT/2)][i%(N/UNROLL_NUM_NTT/2)](2*PRIME_BIT-1, PRIME_BIT);
			}
		}
        for (ap_uint<LOG_RNS+LOG_N+1> i = (RNS_NUM-1)*N; i < (RNS_NUM)*N; i++) {
#pragma HLS PIPELINE
			 out[i] = 0;
		}
    } else if (mode == 1) {
    	for (ap_uint<LOG_RNS+1> r = 0; r < (RNS_NUM-1); r++) {
#pragma HLS PIPELINE
			for (ap_uint<LOG_N+1> i = 0; i < N/2; i++) {
				poly_coeff[r][i/(N/UNROLL_NUM_NTT/2)][i%(N/UNROLL_NUM_NTT/2)] = (in[2*i+1+r*N], in[2*i+r*N]);
			}
		}
		for (ap_uint<LOG_RNS+1> i = 0; i < RNS_NUM-1; i++) {
			NTT(poly_coeff[i], true, MOD[i], Mprime[i], forward_table[i], inv_table[i]);
		}
		for (ap_uint<LOG_RNS+1> r = 0; r < (RNS_NUM-1); r++) {
#pragma HLS PIPELINE
			for (ap_uint<LOG_N+1> i = 0; i < N/2; i++) {
				out[i+r*N]     = poly_coeff[r][i/(N/UNROLL_NUM_NTT/2)][i%(N/UNROLL_NUM_NTT/2)](PRIME_BIT-1, 0);
				out[i+N/2+r*N] = poly_coeff[r][i/(N/UNROLL_NUM_NTT/2)][i%(N/UNROLL_NUM_NTT/2)](2*PRIME_BIT-1, PRIME_BIT);
			}
		}
		for (ap_uint<LOG_RNS+LOG_N+1> i = (RNS_NUM-1)*N; i < (RNS_NUM)*N; i++) {
#pragma HLS PIPELINE
			 out[i] = 0;
		}
    } else if (mode == 2) {
    	for (ap_uint<LOG_RNS+1> r = 0; r < RNS_NUM; r++) {
#pragma HLS PIPELINE
    		for (ap_uint<LOG_N+1> i = 0; i < N/2; i++) {
    			poly_coeff[r][i/(N/UNROLL_NUM_NTT/2)][i%(N/UNROLL_NUM_NTT/2)] = (in[i+N/2+r*N], in[i+r*N]);
    		}
    	}
        for (ap_uint<LOG_RNS+1> i = 0; i < RNS_NUM; i++) {
        	NTT(poly_coeff[i], false, MOD[i+RNS_NUM], Mprime[i+RNS_NUM], forward_table[i+RNS_NUM-1], inv_table[i+RNS_NUM-1]);
        }
        for (ap_uint<LOG_RNS+1> r = 0; r < RNS_NUM; r++) {
        #pragma HLS PIPELINE
			for (ap_uint<LOG_N+1> i = 0; i < N/2; i++) {
				out[2*i+r*N]   = poly_coeff[r][i/(N/UNROLL_NUM_NTT/2)][i%(N/UNROLL_NUM_NTT/2)](PRIME_BIT-1, 0);
				out[2*i+r*N+1] = poly_coeff[r][i/(N/UNROLL_NUM_NTT/2)][i%(N/UNROLL_NUM_NTT/2)](2*PRIME_BIT-1, PRIME_BIT);
			}
		}
    } else if (mode == 3) {
    	for (ap_uint<LOG_RNS+1> r = 0; r < RNS_NUM; r++) {
#pragma HLS PIPELINE
			for (ap_uint<LOG_N+1> i = 0; i < N/2; i++) {
				poly_coeff[r][i/(N/UNROLL_NUM_NTT/2)][i%(N/UNROLL_NUM_NTT/2)] = (in[2*i+1+r*N], in[2*i+r*N]);
			}
		}
		for (ap_uint<LOG_RNS+1> i = 0; i < RNS_NUM; i++) {
			NTT(poly_coeff[i], true, MOD[i+RNS_NUM], Mprime[i+RNS_NUM], forward_table[i+RNS_NUM-1], inv_table[i+RNS_NUM-1]);
		}
		for (ap_uint<LOG_RNS+1> r = 0; r < RNS_NUM; r++) {
#pragma HLS PIPELINE
			for (ap_uint<LOG_N+1> i = 0; i < N/2; i++) {
				out[i+r*N]     = poly_coeff[r][i/(N/UNROLL_NUM_NTT/2)][i%(N/UNROLL_NUM_NTT/2)](PRIME_BIT-1, 0);
				out[i+N/2+r*N] = poly_coeff[r][i/(N/UNROLL_NUM_NTT/2)][i%(N/UNROLL_NUM_NTT/2)](2*PRIME_BIT-1, PRIME_BIT);
			}
		}
    } else {
    	for (ap_uint<LOG_RNS+1> r = 0; r < (RNS_NUM-1); r++) {
#pragma HLS PIPELINE
			for (ap_uint<LOG_N+1> i = 0; i < N/2; i++) {
				poly_coeff[r][i/(N/UNROLL_NUM_NTT/2)][i%(N/UNROLL_NUM_NTT/2)] = (in[i+N/2+which_poly*N], in[i+which_poly*N]);
			}
		}
		for (ap_uint<LOG_RNS+1> i = 0; i < RNS_NUM-1; i++) {
			NTT(poly_coeff[i], false, MOD[i], Mprime[i], forward_table[i], inv_table[i]);
		}
		for (ap_uint<LOG_RNS+1> r = 0; r < (RNS_NUM-1); r++) {
		#pragma HLS PIPELINE
			for (ap_uint<LOG_N+1> i = 0; i < N/2; i++) {
				out[2*i+r*N]   = poly_coeff[r][i/(N/UNROLL_NUM_NTT/2)][i%(N/UNROLL_NUM_NTT/2)](PRIME_BIT-1, 0);
				out[2*i+r*N+1] = poly_coeff[r][i/(N/UNROLL_NUM_NTT/2)][i%(N/UNROLL_NUM_NTT/2)](2*PRIME_BIT-1, PRIME_BIT);
			}
		}
		for (ap_uint<LOG_RNS+LOG_N+1> i = (RNS_NUM-1)*N; i < (RNS_NUM)*N; i++) {
#pragma HLS PIPELINE
			 out[i] = 0;
		}
    }

    return;
}
}
