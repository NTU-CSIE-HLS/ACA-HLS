/**********
Copyright (c) 2018, Xilinx, Inc.
All rights reserved.
Redistribution and use in source and binary forms, with or without modification,
are permitted provided that the following conditions are met:
1. Redistributions of source code must retain the above copyright notice,
this list of conditions and the following disclaimer.
2. Redistributions in binary form must reproduce the above copyright notice,
this list of conditions and the following disclaimer in the documentation
and/or other materials provided with the distribution.
3. Neither the name of the copyright holder nor the names of its contributors
may be used to endorse or promote products derived from this software
without specific prior written permission.
THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO,
THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.
IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE,
EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
**********/

#include <iostream>
#include <fstream>
#include <iomanip>

using namespace std;

#include "help_functions.h"
#include "global.h"


size_t global_work_size[] = {1};
size_t local_work_size [] = {1};


// =========================================
// Helper Function: Loads program to memory
// =========================================
int loadFile2Memory(const char *filename, char **result) {

    int size = 0;

    std::ifstream stream(filename, std::ifstream::binary);
    if (!stream) {
        return -1;
    }

    stream.seekg(0, stream.end);
    size = stream.tellg();
    stream.seekg(0, stream.beg);

    *result = new char[size + 1];
    stream.read(*result, size);
    if (!stream) {
        return -2;
    }
    stream.close();
    (*result)[size] = 0;
    return size;
}

void checkError(int errCode, int line) {

  if (errCode != CL_SUCCESS){
    cout << "Error at line: " << line << ", errCode: "<< errCode << endl;
    // exit(-1);
  }
}


int do_k_add (cl_mem poly1, cl_mem poly2, cl_mem out, int inv) {

  cl_int errCode = 0;
  errCode |= clSetKernelArg(cl_k_add,  0, sizeof(cl_mem), &poly1);
  errCode |= clSetKernelArg(cl_k_add,  1, sizeof(cl_mem), &poly2);
  errCode |= clSetKernelArg(cl_k_add,  2, sizeof(cl_mem), &out);
  errCode |= clSetKernelArg(cl_k_add,  3, sizeof(int), &inv);
  errCode |= clEnqueueNDRangeKernel (Command_Queue, cl_k_add, 1, NULL,
                                     global_work_size, local_work_size, 0, NULL, NULL);

  return errCode;
}

int do_k_mulWise (cl_mem poly1, cl_mem poly2, cl_mem out, int inv) {

  cl_int errCode = 0;
  errCode |= clSetKernelArg(cl_k_mulWise,  0, sizeof(cl_mem), &poly1);
  errCode |= clSetKernelArg(cl_k_mulWise,  1, sizeof(cl_mem), &poly2);
  errCode |= clSetKernelArg(cl_k_mulWise,  2, sizeof(cl_mem), &out);
  errCode |= clSetKernelArg(cl_k_mulWise,  3, sizeof(int), &inv);
  errCode |= clEnqueueNDRangeKernel (Command_Queue, cl_k_mulWise, 1, NULL,
                                     global_work_size, local_work_size, 0, NULL, NULL);

  return errCode;
}

int do_k_mulConst (cl_mem in, cl_mem out, int const_val, int inv) {

  cl_int errCode = 0;
  errCode |= clSetKernelArg(cl_k_mulConst,  0, sizeof(cl_mem), &in);
  errCode |= clSetKernelArg(cl_k_mulConst,  1, sizeof(cl_mem), &out);
  errCode |= clSetKernelArg(cl_k_mulConst,  2, sizeof(int), &const_val);
  errCode |= clSetKernelArg(cl_k_mulConst,  3, sizeof(int), &inv);
  errCode |= clEnqueueNDRangeKernel (Command_Queue, cl_k_mulConst, 1, NULL,
                                     global_work_size, local_work_size, 0, NULL, NULL);

  return errCode;
}

int do_k_mov (cl_mem in, cl_mem out) {
  
  cl_int errCode = 0;
  errCode |= clSetKernelArg(cl_k_mov,  0, sizeof(cl_mem), &in);
  errCode |= clSetKernelArg(cl_k_mov,  1, sizeof(cl_mem), &out);
  errCode |= clEnqueueNDRangeKernel (Command_Queue, cl_k_mov, 1, NULL,
                                     global_work_size, local_work_size, 0, NULL, NULL);

  return errCode;
}

int do_k_apply_galois (cl_mem in, int r, cl_mem out) {
  
  cl_int errCode = 0;
  errCode |= clSetKernelArg(cl_k_apply_galois,  0, sizeof(cl_mem), &in);
  errCode |= clSetKernelArg(cl_k_apply_galois,  1,    sizeof(int), &r);
  errCode |= clSetKernelArg(cl_k_apply_galois,  2, sizeof(cl_mem), &out);
  errCode |= clEnqueueNDRangeKernel (Command_Queue, cl_k_apply_galois, 1, NULL,
                                     global_work_size, local_work_size, 0, NULL, NULL);

  return errCode;

}


int do_k_ntt (cl_mem in, cl_mem out, int mode, int which_poly) {

  cl_int errCode = 0;
  errCode |= clSetKernelArg(cl_k_ntt,  0, sizeof(cl_mem), &in);
  errCode |= clSetKernelArg(cl_k_ntt,  1, sizeof(cl_mem), &out);
  errCode |= clSetKernelArg(cl_k_ntt,  2, sizeof(int), &mode);
  errCode |= clSetKernelArg(cl_k_ntt,  3, sizeof(int), &which_poly);
  errCode |= clEnqueueNDRangeKernel (Command_Queue, cl_k_ntt, 1, NULL,
                                     global_work_size, local_work_size, 0, NULL, NULL);

  return errCode;

}

int do_k_scale (cl_mem in1, cl_mem in2, cl_mem out, int which) {

  cl_int errCode = 0;
  errCode |= clSetKernelArg(cl_k_scale,  0, sizeof(cl_mem), &in1);
  errCode |= clSetKernelArg(cl_k_scale,  1, sizeof(cl_mem), &in2);
  errCode |= clSetKernelArg(cl_k_scale,  2, sizeof(cl_mem), &out);
  errCode |= clSetKernelArg(cl_k_scale,  3, sizeof(int), &which);
  errCode |= clEnqueueNDRangeKernel (Command_Queue, cl_k_scale, 1, NULL,
                                     global_work_size, local_work_size, 0, NULL, NULL);

  return errCode;

}

int do_k_lift (cl_mem in1, cl_mem out, int which) {

  cl_int errCode = 0;
  errCode |= clSetKernelArg(cl_k_lift,  0, sizeof(cl_mem), &in1);
  errCode |= clSetKernelArg(cl_k_lift,  1, sizeof(cl_mem), &out);
  errCode |= clSetKernelArg(cl_k_lift,  2, sizeof(int), &which);
  errCode |= clEnqueueNDRangeKernel (Command_Queue, cl_k_lift, 1, NULL,
                                     global_work_size, local_work_size, 0, NULL, NULL);

  return errCode;

}

int do_k_mulInv (cl_mem in, cl_mem out, int inv) {
  cl_int errCode = 0;
  errCode |= clSetKernelArg(cl_k_mulInv,  0, sizeof(cl_mem), &in);
  errCode |= clSetKernelArg(cl_k_mulInv,  1, sizeof(cl_mem), &out);
  errCode |= clSetKernelArg(cl_k_mulInv,  2, sizeof(int), &inv);
  errCode |= clEnqueueNDRangeKernel (Command_Queue, cl_k_mulInv, 1, NULL,
                                     global_work_size, local_work_size, 0, NULL, NULL);

  return errCode;
}
