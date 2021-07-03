#include <CL/cl.h>

#define UNIT_INT_COUNT 8*8192

extern cl_command_queue Command_Queue;
extern cl_context       Context;

extern cl_kernel cl_k_ntt;
extern cl_kernel cl_k_add;
extern cl_kernel cl_k_mulWise;
extern cl_kernel cl_k_mulConst;
extern cl_kernel cl_k_mov;
extern cl_kernel cl_k_apply_galois;
extern cl_kernel cl_k_scale;
extern cl_kernel cl_k_lift;
extern cl_kernel cl_k_mulInv;

extern cl_mem GM_dense100[13][4];
extern cl_mem GM_mask[4];
extern cl_mem GM_dense10[4];
extern cl_mem GM_bias100[4];

extern cl_mem GM_rlk[4][7][2];
extern cl_mem GM_ct[25][4][2];
extern cl_mem GM_gk[4][25][7][2];

extern cl_mem temp1[2];
extern cl_mem temp2[2];
extern cl_mem temp3[3];
extern cl_mem temp4[3];
extern cl_mem temp5[2];
extern cl_mem temp6[7];
extern cl_mem temp7[7];
extern cl_mem temp8[7];
extern cl_mem temp9[2];
extern cl_mem temp10[2];
extern cl_mem temp_result[5][25][4][2];

