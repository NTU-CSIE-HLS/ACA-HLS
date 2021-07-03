#ifndef __MOD_H__
#define __MOD_H__

#include "ap_int.h"

extern "C" {

#define CENTER_SHIFT 1 // 1:shift to [-p/2, p/2] , 0: original [0, p-1] //TODO
#define FUNCTION_SIGNED_OR_NOT 1 // 1:signed 0:unsigned //TODO
#define PRIME_SETTING 0 //TODO


#if PRIME_SETTING == 0
//default prime list

#define N_PRIME 16
#define WINDOW_WIDTH 6
#define P_WIDTH 32
#define MODULAR_TABLE_SIZE (1 << WINDOW_WIDTH)

static ap_uint<P_WIDTH> PRIME_LIST[N_PRIME] = \
 { 2147565569, 2148155393, 2148384769, 2148728833, 2148794369, 2149072897, 2149171201, 0, 2149466113, 2149662721, 2149810177, 2150072321, 2150301697, 2150318081, 2150563841, 2150612993 };


////////////////////////// put the above in mod.h

#endif //PRIME_SETTING



#define MODULAR_TABLE_POWER (P_WIDTH)
#define XY_WIDTH (2*P_WIDTH)


ap_uint<P_WIDTH> reduction_unsigned(ap_uint<XY_WIDTH> xy, ap_uint<P_WIDTH> p, unsigned int idx_p);

#if FUNCTION_SIGNED_OR_NOT == 1
ap_int<P_WIDTH> modular_multiplication(ap_int<P_WIDTH> x, ap_int<P_WIDTH> y, ap_uint<P_WIDTH> p, unsigned int idx_p);
ap_int<P_WIDTH> reduction(ap_int<XY_WIDTH> xy, ap_uint<P_WIDTH> p, unsigned int idx_p);

#elif FUNCTION_SIGNED_OR_NOT == 0
ap_uint<P_WIDTH> modular_multiplication(ap_uint<P_WIDTH> x, ap_uint<P_WIDTH> y, ap_uint<P_WIDTH> p, unsigned int idx_p);
#endif //FUNCTION_SIGNED_OR_NOT


#endif //ifndef __MOD_H__

}
