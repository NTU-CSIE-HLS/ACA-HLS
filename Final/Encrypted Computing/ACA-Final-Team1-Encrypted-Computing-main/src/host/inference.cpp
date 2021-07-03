#include <math.h>
#include <stdlib.h>
#include "help_functions.h"
#include "model_weight.h"
#include <vector>
#include <iostream>
#include "global.h"

using namespace std;

#define N 8192


#define INVERSE   0
#define FORWARD   1
#define SPECIAL   2
#define LIFT_TO_P false
#define LIFT_TO_Q true
#define FIRST     false
#define SECOND    true


int power(long long x, unsigned int y, int p)
{
    int res = 1; 
    x = x % p; 
    if (x == 0) return 0; 
    while (y > 0)
    {
        if (y & 1)
            res = (res*x) % p;
        y = y>>1; 
        x = (x*x) % p;
    }
    return res;
}

void keyswitching(cl_mem ct2,
                  cl_mem k[7][2],
                  cl_mem out[2])
{
    for (int i = 0; i < 7; i++) {
        do_k_NTT(ct2, temp6[i], SPECIAL, FIRST, i);
        do_k_mulWise(temp6[i], k[i][0], temp7[i]);
        do_k_mulWise(temp6[i], k[i][1], temp8[i]);
    }
    for (int i = 1; i < 7; i++) {
        do_k_add(temp7[0], temp7[i], temp7[0]);
        do_k_add(temp8[0], temp8[i], temp8[0]);
    }
    do_k_mov(temp7[0], out[0]);
    do_k_mov(temp8[0], out[1]);
    do_k_mov(ct2, out[0]);
    do_k_mov(ct2, out[1]);
}

// HE primitives
void MulConst(cl_mem in[4][2],
              cl_mem out[4][2],
              int constant) 
{
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 2; j++) {
            do_k_mulConst(in[i][j], out[i][j], constant, 0);
        }
    }
}

void MulWise(cl_mem in1[4][2],
             cl_mem in2[4],
             cl_mem out[4][2]) 
{
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 2; j++) {
            do_k_mulWise(in1[i][j], in2[i], out[i][j], 0);
        }
    }
}

void HMul(cl_mem in1[4][2],
          cl_mem in2[4][2],
          cl_mem out[4][2]) 

{
    for (int i = 0; i < 4; i++) {
        do_k_NTT(in1[i][0], in1[i][0], INVERSE, FIRST, 0);
        do_k_NTT(in2[i][0], in2[i][0], INVERSE, FIRST, 0);
        do_k_NTT(in1[i][1], in1[i][1], INVERSE, FIRST, 0);
        do_k_NTT(in2[i][1], in2[i][1], INVERSE, FIRST, 0);

        do_k_lift(in1[i][0], temp1[0], LIFT_TO_P);
        do_k_lift(in2[i][0], temp2[0], LIFT_TO_P);
        do_k_lift(in1[i][1], temp1[1], LIFT_TO_P);
        do_k_lift(in2[i][1], temp2[1], LIFT_TO_P);

        do_k_NTT(in1[i][0], in1[i][0], FORWARD, FIRST, 0);
        do_k_NTT(in2[i][0], in2[i][0], FORWARD, FIRST, 0);
        do_k_NTT(in1[i][1], in1[i][1], FORWARD, FIRST, 0);
        do_k_NTT(in2[i][1], in2[i][1], FORWARD, FIRST, 0);
        do_k_NTT(temp1[0], temp1[0], FORWARD, SECOND, 0);
        do_k_NTT(temp2[0], temp2[0], FORWARD, SECOND, 0);
        do_k_NTT(temp1[1], temp1[1], FORWARD, SECOND, 0);
        do_k_NTT(temp2[1], temp2[1], FORWARD, SECOND, 0);

        // Tensoring
        do_k_mulWise(in1[i][0], in2[i][0], temp3[0], 0);
        do_k_mulWise(in1[i][1], in2[i][0], temp3[1], 0);
        do_k_mulWise(in1[i][0], in2[i][1], temp3[3], 0);
        do_k_mulWise(in1[i][1], in2[i][1], temp3[2], 0);
        do_k_mulWise(temp1[0], temp2[0], temp4[0], 1);
        do_k_mulWise(temp1[1], temp2[0], temp4[1], 1);
        do_k_mulWise(temp1[0], temp2[1], temp4[3], 1);
        do_k_mulWise(temp1[1], temp2[1], temp4[2], 1);
        do_k_add(temp3[1], temp3[3], temp3[1], 0);
        do_k_add(temp4[1], temp4[3], temp4[1], 1);

        do_k_NTT(temp3[0], temp3[0], INVERSE, FIRST, 0);
        do_k_NTT(temp3[1], temp3[1], INVERSE, FIRST, 0);
        do_k_NTT(temp3[2], temp3[2], INVERSE, FIRST, 0);
        do_k_NTT(temp4[0], temp4[0], INVERSE, SECOND, 0);
        do_k_NTT(temp4[1], temp4[1], INVERSE, SECOND, 0);
        do_k_NTT(temp4[2], temp4[2], INVERSE, SECOND, 0);

        do_k_scale(temp3[0], temp4[0], temp3[0]);
        do_k_scale(temp3[1], temp4[1], temp3[1]);
        do_k_scale(temp3[2], temp4[2], temp3[2]);

        do_k_lift(temp3[0], temp3[0], LIFT_TO_Q);
        do_k_lift(temp3[1], temp3[1], LIFT_TO_Q);
        do_k_lift(temp3[2], temp3[2], LIFT_TO_Q);

        // relin
        do_k_NTT(temp3[0], temp3[0], FORWARD, FIRST, 0);
        do_k_NTT(temp3[1], temp3[1], FORWARD, FIRST, 0);

        keyswitching(temp3[2], GM_rlk[i], temp5);
        
        do_k_add(temp3[0], temp5[0], out[i][0], 0);
        do_k_add(temp3[1], temp5[1], out[i][1], 0);
    }
}

void HAdd(cl_mem in1[4][2],
          cl_mem in2[4][2],
          cl_mem out[4][2]) 
{
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 2; j++) {
            do_k_add(in1[i][j], in2[i][j], out[i][j], 0);
        }
    }
}

void rotate_row(cl_mem in1[4][2],
                int r,
                cl_mem out[4][2]) 
{
    for (int i = 0; i < 4; i++) {
        do_k_NTT(in1[i][0], in1[i][0], INVERSE, FIRST, 0);
        do_k_NTT(in1[i][1], in1[i][1], INVERSE, FIRST, 0);
        do_k_apply_galois(in1[i][0], 2*N-1, in1[i][0]);
        do_k_apply_galois(in1[i][1], 2*N-1, in1[i][1]);
        keyswitching(in1[i][1], GM_gk[i][24], temp9);
        do_k_mov(temp9[1], out[i][1]);
        do_k_add(in1[i][0], temp9[0], out[i][0], 0);
        do_k_NTT(out[i][0], out[i][0], FORWARD, FIRST, 0);
        do_k_NTT(out[i][1], out[i][1], FORWARD, FIRST, 0);
    }
}

void rotate_column(cl_mem in1[4][2],
                   int r, // +- 1, 2, 4, ...
                   cl_mem out[4][2]) {
    vector<int> rot;
    vector<int> which_key;
    r = -r;
    while (r != 0) {
        double temp = log2(abs(r));
        int t = round(temp);
        int num = 1<<t;
        if (r < 0) {
            num = -num;
            t   = t + ((int)(log2(N))-1);
        }
        r -= num;
        if (num % (N/2) == 0)
            continue;
        num = (num + N/2) % (N/2);
        num = power(3,num,N);
        rot.push_back(num);
        which_key.push_back(t);
    }

    for (int a = 0; a < rot.size(); a++) {
        for (int i = 0; i < 4; i++) {
            do_k_NTT(in1[i][0], in1[i][0], INVERSE, FIRST, 0);
            do_k_NTT(in1[i][1], in1[i][1], INVERSE, FIRST, 0);
            do_k_apply_galois(in1[i][0], rot[a], in1[i][0]);
            do_k_apply_galois(in1[i][1], rot[a], in1[i][1]);
            keyswitching(in1[i][1], GM_gk[i][which_key[a]], temp10);
            do_k_mov(temp10[1], out[i][1]);
            do_k_add(in1[i][0], temp10[0], out[i][0], 0);
            do_k_NTT(out[i][0], out[i][0], FORWARD, FIRST, 0);
            do_k_NTT(out[i][1], out[i][1], FORWARD, FIRST, 0);
        }
    }
}

void CNN(cl_mem in1[25][4][2], cl_mem out[4][2]) { // temp
    for (int c = 0; c < 5; c++) {
        for (int i = 0; i < 25; i++) {
            MulConst(in1[i], temp_result[c][i], cnn_weight[c][i]);

        }
    }
    for (int c = 0; c < 5; c++)
        for (int i = 1; i < 25; i++)
            HAdd(temp_result[c][0], temp_result[c][i], temp_result[c][0]);
    
    for (int i = 1; i < 5; i++)
        rotate_column(temp_result[i][0], 169*i, temp_result[i][0]);
    for (int i = 1; i < 5; i++)
        HAdd(temp_result[0][0], temp_result[i][0], temp_result[0][0]);

    for (int i = 0; i < 4; ++i) {
      for (int j = 0; j < 2; ++j){
        do_k_mov(temp_result[0][0][i][j], out[i][j]);
      }
    }
    do_k_mov(temp_result[0][0], out[0]);
    do_k_mov(temp_result[0][1], out[1]);

}

void Square(cl_mem in1[2]) {
    HMul(in1, in1, in1);
}

void Dense100(cl_mem in1[2]) { // rt_2[2], rt_4[2], rt_8[2], temp[13]
    rotate_column(in1, 1024, rt_2);
    HAdd(in1, rt_2, rt_2);
    rotate_column(rt_2, 2048, rt_4);
    HAdd(rt_2, rt_4, rt_4);
    rotate_row(rt_4, 1, rt_8);
    HAdd(rt_4, rt_8, rt_8);

    for (int i = 0; i < 13; i++) {
        MulWise(in1)
    }
}

void Dense10() {
}

// inference.cpp
void inference() 
{
    clCreateCommandQueue(context, device_id, CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE, &err);
    CNN     (GM_ct, GM_ct[0]);
    clEnqueueBarrierWithWaitList(Command_Queue, 0, NULL, NULL);
    Square  (GM_ct);    
    clEnqueueBarrierWithWaitList(Command_Queue, 0, NULL, NULL);
    Dense100(GM_ct);
    clEnqueueBarrierWithWaitList(Command_Queue, 0, NULL, NULL);
    Square  (GM_ct);
    clEnqueueBarrierWithWaitList(Command_Queue, 0, NULL, NULL);
    Dense10 (GM_ct);
    clEnqueueBarrierWithWaitList(Command_Queue, 0, NULL, NULL);
}

void inference_test() 
{

    do_k_add( GM_ct[0][0][0],  GM_ct[0][0][0],  GM_ct[0][0][1], 1);
    do_k_mulConst(GM_ct[0][0][0], GM_ct[0][0][1], 1<<17, 1);
    do_k_mulWise( GM_ct[0][0][0],  GM_ct[0][0][0],  GM_ct[0][0][1], 1);
    do_k_mov(GM_ct[0][0][0], GM_ct[0][0][1]);

   do_k_ntt(         GM_ct[0][0][0], temp_result[0][0][0][0], 4, 2);
   do_k_ntt(temp_result[0][0][0][0],          GM_ct[0][0][1], 1, 0);
   do_k_mulInv(GM_ct[0][0][1], GM_ct[0][0][1], 1);

	do_k_lift(GM_ct[0][0][0], temp_result[0][0][0][0], 0);
	do_k_lift(temp_result[0][0][0][0], GM_ct[0][0][1], 1);

	do_k_scale(GM_ct[0][0][0], GM_ct[0][0][0], GM_ct[0][0][1], 0);


	do_k_apply_galois(         GM_ct[0][0][0],   243, temp_result[0][0][0][0]);
	do_k_apply_galois(temp_result[0][0][0][0],  6203, temp_result[0][0][0][1]);
	do_k_apply_galois(temp_result[0][0][0][1], 16383, temp_result[0][0][1][0]);
	do_k_apply_galois(temp_result[0][0][1][0], 16383,          GM_ct[0][0][1]);
}

//
