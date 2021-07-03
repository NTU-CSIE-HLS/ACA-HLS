#include <CL/cl.h>

#define NUM_KERNEL 7
#define DEVICE_MEM_UNIT 8*8192

cl_command_queue Command_Queue;

cl_kernel K_HE_interface[NUM_KERNEL];

// dense100_size   = 13 * 4 * (8 * 8192);
// mask_size       = 4 *      (8 * 8192);
// dense10_size    = 4 *      (8 * 8192);
// bias100_size    = 4 *      (8 * 8192);

cl_mem GM_dense100[13][4];
cl_mem GM_mask[4];
cl_mem GM_dense10[4];
cl_mem GM_bias100[4];

// rlk_size = 4  * 7  * 2 *     (8 * 8192);
//  ct_size = 25 * 4  * 2 *     (8 * 8192);
//  gk_size = 4  * 25 * 7 * 2 * (8 * 8192); 

cl_mem GM_rlk[4][7][2];
cl_mem GM_ct[25][4][2];
cl_mem GM_gk[4][25][7][2];

