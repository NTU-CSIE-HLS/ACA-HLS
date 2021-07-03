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

#include <CL/cl.h>
int loadFile2Memory(const char *filename,
                    char **result);
void checkError(int errCode, int line);
int do_k_add (cl_mem poly1, cl_mem poly2, cl_mem out, int inv);
int do_k_mulWise (cl_mem poly1, cl_mem poly2, cl_mem out, int inv);
int do_k_mulConst (cl_mem in, cl_mem out, int const_val, int inv) ;
int do_k_mov (cl_mem in, cl_mem out) ;
int do_k_apply_galois (cl_mem in, int r, cl_mem out);
int do_k_ntt (cl_mem in, cl_mem out, int mode, int which_poly);
int do_k_scale (cl_mem in1, cl_mem in2, cl_mem out, int which);
int do_k_lift (cl_mem in1, cl_mem out, int which);
int do_k_mulInv (cl_mem in, cl_mem out, int inv);
