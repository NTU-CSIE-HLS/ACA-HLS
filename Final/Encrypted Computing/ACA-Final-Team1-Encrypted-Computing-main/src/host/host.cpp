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

/*******************************************************************************
** HOST Code
*******************************************************************************/

#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <stdlib.h>
#include <stdio.h>
#include <cstring>
#include <iostream>
#include <fstream>
#include <iomanip>

using namespace std;

#include <CL/cl.h>

#include "help_functions.h"
#include "global.h"
#include "model_weight.h"
#include "sockpp/tcp_acceptor.h"
#include "sockpp/version.h"
#include "inference.h"

// Global Variables
cl_context          Context;
cl_command_queue    Command_Queue;

cl_kernel cl_k_add;
cl_kernel cl_k_mulWise;
cl_kernel cl_k_mulConst;
cl_kernel cl_k_mov;
cl_kernel cl_k_apply_galois;
cl_kernel cl_k_ntt;
cl_kernel cl_k_scale;
cl_kernel cl_k_lift;
cl_kernel cl_k_mulInv;

cl_mem GM_dense100[13][4];
cl_mem GM_mask[4];
cl_mem GM_dense10[4];
cl_mem GM_bias100[4];

cl_mem GM_rlk[4][7][2];
cl_mem GM_ct[25][4][2];
cl_mem GM_gk[4][25][7][2];

cl_mem temp1[2];
cl_mem temp2[2];
cl_mem temp3[3];
cl_mem temp4[3];
cl_mem temp5[2];
cl_mem temp6[7];
cl_mem temp7[7];
cl_mem temp8[7];
cl_mem temp9[2];
cl_mem temp10[2];
cl_mem temp_result[5][25][4][2];

#define ALL_MESSAGES

// ********************************************************************************** //
// ---------------------------------------------------------------------------------- //
//                          M A I N    F U N C T I O N                                //
// ---------------------------------------------------------------------------------- //
// ********************************************************************************** //

int main(int argc, char* argv[])
{
  cout << endl;


  // ============================================================================
  // Step 1: Check Command Line Arguments
  // ============================================================================
  //    o) argv[1] Platfrom Vendor
  //    o) argv[2] Device Name
  //    o) argv[3] XCLBIN file
  // ============================================================================
  #ifdef ALL_MESSAGES
  cout << "HOST-Info: ============================================================= " << endl;
  cout << "HOST-Info: (Step 1) Check Command Line Arguments                      " << endl;
  cout << "HOST-Info: ============================================================= " << endl;
  #endif

  if (argc != 2)
  {
    cout << "HOST-Error: Incorrect command line syntax " << endl;
    cout << "HOST-Info:  Usage: " << argv[0] << " <XCLBIN_File>" << endl << endl;
    return EXIT_FAILURE;
  } 

  const char* Target_Platform_Vendor   = "Xilinx";
  const char* Target_Device_Name       = "xilinx_u50_gen3x16_xdma_201920_3";
  const char* xclbinFilename           = argv[1];
  cout << "HOST-Info: Platform_Vendor   : " << Target_Platform_Vendor << endl;
  cout << "HOST-Info: Device_Name       : " << Target_Device_Name << endl;
  cout << "HOST-Info: XCLBIN_file       : " << xclbinFilename << endl;


  // ============================================================================
  // Step 2: Detect Target Platform and Target Device in a system. 
  //         Create Context and Command Queue.
  // ============================================================================
  // Variables:
  //   o) Target_Platform_Vendor[] - defined as main() input argument 
  //   o) Target_Device_Name[]     - defined as main() input argument
  // 
  // After that
  //   o) Create a Context
  //   o) Create a Command Queue
  // ============================================================================
  cout << endl;
  #ifdef ALL_MESSAGES
  cout << "HOST-Info: ============================================================= " << endl;
  cout << "HOST-Info: (Step 2) Detect Target Platform and Target Device in a system " << endl;
  cout << "HOST-Info:          Create Context and Command Queue                     " << endl;
  cout << "HOST-Info: ============================================================= " << endl;
  #endif

  cl_uint         ui;

  cl_platform_id      *Platform_IDs;
  cl_uint             Nb_Of_Platforms;
  cl_platform_id      Target_Platform_ID;
  bool                Platform_Detected;
  char                *platform_info;

  cl_device_id        *Device_IDs;
  cl_uint             Nb_Of_Devices;
  cl_device_id        Target_Device_ID;
  bool                Device_Detected;
  char                *device_info;

  cl_int              errCode;
  size_t              size;

  // ------------------------------------------------------------------------------------
  // Step 2.1: Get All PLATFORMS, then search for Target_Platform_Vendor (CL_PLATFORM_VENDOR)
  // ------------------------------------------------------------------------------------
  
  // Get the number of platforms
  // ..................................................
  errCode = clGetPlatformIDs(0, NULL, &Nb_Of_Platforms);
  if (errCode != CL_SUCCESS || Nb_Of_Platforms <= 0) {
    cout << endl << "HOST-Error: Failed to get the number of available platforms" << endl << endl;
    return EXIT_FAILURE;
  }

  #ifdef ALL_MESSAGES
  cout << "HOST-Info: Number of detected platforms : " << Nb_Of_Platforms << endl;
  #endif

  // Allocate memory to store platforms
  // ..................................................
  Platform_IDs = new cl_platform_id[Nb_Of_Platforms];
  if (!Platform_IDs) {
    cout << endl << "HOST-Error: Out of Memory during memory allocation for Platform_IDs" << endl << endl;
    return EXIT_FAILURE;
  }

  // Get and store all PLATFORMS
  // ..................................................
  errCode = clGetPlatformIDs(Nb_Of_Platforms, Platform_IDs, NULL);
  if (errCode != CL_SUCCESS) {
    cout << endl << "HOST-Error: Failed to get the available platforms" << endl << endl;
    return EXIT_FAILURE;
  }

  // Search for Platform (ex: Xilinx) using: CL_PLATFORM_VENDOR = Target_Platform_Vendor
  // ....................................................................................
  Platform_Detected = false;
  for (ui = 0; ui < Nb_Of_Platforms; ui++) {

    errCode = clGetPlatformInfo(Platform_IDs[ui], CL_PLATFORM_VENDOR, 0, NULL, &size);
    if (errCode != CL_SUCCESS) {
      cout << endl << "HOST-Error: Failed to get the size of the Platofrm parameter " << "CL_PLATFORM_VENDOR" << " value " << endl << endl;
      return EXIT_FAILURE;
    }

    platform_info = new char[size];
    if (!platform_info) {
      cout << endl << "HOST-Error: Out of Memory during memory allocation for Platform Parameter " << "CL_PLATFORM_VENDOR" << endl << endl;
      return EXIT_FAILURE;
    }

    errCode = clGetPlatformInfo(Platform_IDs[ui], CL_PLATFORM_VENDOR, size, platform_info , NULL);
    if (errCode != CL_SUCCESS) {
      cout << endl << "HOST-Error: Failed to get the " << "CL_PLATFORM_VENDOR" << " platform info" << endl << endl;
      return EXIT_FAILURE;
    }

    // Check if the current platform matches Target_Platform_Vendor
    // .............................................................
    if (strcmp(platform_info, Target_Platform_Vendor) == 0) {
      Platform_Detected        = true;
      Target_Platform_ID       = Platform_IDs[ui];
      #ifdef ALL_MESSAGES
      cout << "HOST-Info: Selected platform            : " << Target_Platform_Vendor << endl << endl;
      #endif
    }
  }

  if (Platform_Detected == false) {
    cout << endl << "HOST-Error: Failed to get detect " << Target_Platform_Vendor << " platform" << endl << endl;
    return EXIT_FAILURE;
  }


  // ------------------------------------------------------------------------------------
  // Step 2.2:  Get All Devices for selected platform Target_Platform_ID
  //            then search for Xilinx platform (CL_DEVICE_NAME = Target_Device_Name)
  // ------------------------------------------------------------------------------------

  // Get the Number of Devices
  // ............................................................................
  errCode = clGetDeviceIDs(Target_Platform_ID, CL_DEVICE_TYPE_ALL, 0, NULL, &Nb_Of_Devices);
  if (errCode != CL_SUCCESS) {
    cout << endl << "HOST-Error: Failed to get the number of available Devices" << endl << endl;
    return EXIT_FAILURE;
  }
  #ifdef ALL_MESSAGES
  cout << "HOST-Info: Number of available devices  : " << Nb_Of_Devices << endl;
  #endif

  Device_IDs = new cl_device_id[Nb_Of_Devices];
  if (!Device_IDs) {
    cout << endl << "HOST-Error: Out of Memory during memory allocation for Device_IDs" << endl << endl;
    return EXIT_FAILURE;
  }

  errCode = clGetDeviceIDs(Target_Platform_ID, CL_DEVICE_TYPE_ALL, Nb_Of_Devices, Device_IDs, NULL);
  if (errCode != CL_SUCCESS) {
    cout << endl << "HOST-Error: Failed to get available Devices" << endl << endl;
    return EXIT_FAILURE;
  }

  // Search for CL_DEVICE_NAME = Target_Device_Name
  // ............................................................................
  Device_Detected = false;
  for (ui = 0; ui < Nb_Of_Devices; ui++) {
    errCode = clGetDeviceInfo(Device_IDs[ui], CL_DEVICE_NAME, 0, NULL, &size);
    if (errCode != CL_SUCCESS) {
      cout << endl << "HOST-Error: Failed to get the size of the Device parameter value " << "CL_DEVICE_NAME" << endl << endl;
      return EXIT_FAILURE;
    }

    device_info = new char[size];
    if (!device_info) {
      cout << endl << "HOST-Error: Out of Memory during memory allocation for Device parameter " << "CL_DEVICE_NAME" << " value " << endl << endl;
      return EXIT_FAILURE;
    }

    errCode = clGetDeviceInfo(Device_IDs[ui], CL_DEVICE_NAME, size, device_info, NULL);
    if (errCode != CL_SUCCESS) {
      cout << endl << "HOST-Error: Failed to get the " << "CL_DEVICE_NAME" << " device info" << endl << endl;
      return EXIT_FAILURE;
    }

    // Check if the current device matches Target_Device_Name
    // ............................................................................
    if (strcmp(device_info, Target_Device_Name) == 0) {
      Device_Detected        = true;
      Target_Device_ID       = Device_IDs[ui];
    }
  }

  if (Device_Detected == false) {
    cout << endl << "HOST-Error: Failed to get detect " << Target_Device_Name << " device" << endl << endl;
    return EXIT_FAILURE;
  } else {
    #ifdef ALL_MESSAGES
    cout << "HOST-Info: Selected device              : " << Target_Device_Name << endl << endl;
    #endif
  }

  // ------------------------------------------------------------------------------------
  // Step 2.3: Create Context
  // ------------------------------------------------------------------------------------
  #ifdef ALL_MESSAGES
  cout << "HOST-Info: Creating Context ... " << endl;
  #endif
  Context = clCreateContext(0, 1, &Target_Device_ID, NULL, NULL, &errCode);
  if (errCode != CL_SUCCESS) {
    cout << endl << "HOST-Error: Failed to create a Context" << endl << endl;
    return EXIT_FAILURE;
  }

  // ------------------------------------------------------------------------------------
  // Step 2.4: Create Command Queue (commands are executed in-order)
  // ------------------------------------------------------------------------------------
  #ifdef ALL_MESSAGES
  cout << "HOST-Info: Creating Command Queue ... " << endl;
  #endif
  Command_Queue = clCreateCommandQueue(Context, Target_Device_ID, CL_QUEUE_PROFILING_ENABLE, &errCode);
  if (errCode != CL_SUCCESS) {
    cout << endl << "HOST-Error: Failed to create a Command Queue" << endl << endl;
    return EXIT_FAILURE;
  }

  // ============================================================================
  // Step 3: Create Program and Kernel
  // ============================================================================
  //   o) Create a Program from a Binary File and Build it
  //   o) Create a Kernel
  // ============================================================================
  #ifdef ALL_MESSAGES
  cout << endl;
  cout << "HOST-Info: ============================================================= " << endl;
  cout << "HOST-Info: (Step 3) Create Program and Kernels                           " << endl;
  cout << "HOST-Info: ============================================================= " << endl;
  #endif

  // ------------------------------------------------------------------
  // Step 3.1: Load Binary File from a disk to Memory
  // ------------------------------------------------------------------
  unsigned char *xclbin_Memory;
  int program_length;
  
  #ifdef ALL_MESSAGES
  cout << "HOST-Info: Loading " << xclbinFilename << " binary file to memory ..." << endl;
  #endif

  program_length = loadFile2Memory(xclbinFilename, (char **) &xclbin_Memory);
  if (program_length < 0) {
    cout << endl << "HOST-Error: Failed to load " << xclbinFilename << " binary file to memory" << endl << endl;
    return EXIT_FAILURE;
  }

  // ------------------------------------------------------------
  // Step 3.2: Create a program using a Binary File
  // ------------------------------------------------------------
  size_t     Program_Length_in_Bytes;
  cl_program Program;
  cl_int     Binary_Status;
  
  #ifdef ALL_MESSAGES
  cout << "HOST-Info: Creating Program with Binary ..." << endl;
  #endif
  Program_Length_in_Bytes = program_length;
  Program = clCreateProgramWithBinary(Context, 1, &Target_Device_ID, &Program_Length_in_Bytes, 
    (const unsigned char **) &xclbin_Memory, &Binary_Status, &errCode);
  if (errCode != CL_SUCCESS) {
    cout << endl << "HOST-Error: Failed to create a Program from a Binary" << endl << endl;
    return EXIT_FAILURE;
  }

  // ----------------------------------------------------------------------
  // Step 3.3: Build (compiles and links) a program executable from binary
  // ----------------------------------------------------------------------
  #ifdef ALL_MESSAGES
  cout << "HOST-Info: Building the Program ..." << endl;
  #endif

  errCode = clBuildProgram(Program, 1, &Target_Device_ID, NULL, NULL, NULL);
  if (errCode != CL_SUCCESS) {
    cout << endl << "HOST-Error: Failed to build a Program Executable" << endl << endl;
    return EXIT_FAILURE;
  }

  // ADD //
  // -------------------------------------------------------------
  // Step 3.4: Create a Kernels
  // -------------------------------------------------------------

  #ifdef ALL_MESSAGES
  cout << "HOST-Info: Creating a Kernel: KVConstAdd ..." << endl;
  #endif
  cl_k_add = clCreateKernel(Program, "K_add", &errCode);
  if (errCode != CL_SUCCESS) {
    cout << endl << "HOST-Error: Failed to create K_KVConstAdd" << endl << endl;
    return EXIT_FAILURE;
  }

  #ifdef ALL_MESSAGES
  cout << "HOST-Info: Creating a Kernel: KVConstAdd ..." << endl;
  #endif
  cl_k_mulWise = clCreateKernel(Program, "K_mulWise", &errCode);
  if (errCode != CL_SUCCESS) {
    cout << endl << "HOST-Error: Failed to create K_KVConstAdd" << endl << endl;
    return EXIT_FAILURE;
  }

  #ifdef ALL_MESSAGES
  cout << "HOST-Info: Creating a Kernel: KVConstAdd ..." << endl;
  #endif
  cl_k_mulConst = clCreateKernel(Program, "K_mulConst", &errCode);
  if (errCode != CL_SUCCESS) {
    cout << endl << "HOST-Error: Failed to create K_KVConstAdd" << endl << endl;
    return EXIT_FAILURE;
  }

  #ifdef ALL_MESSAGES
  cout << "HOST-Info: Creating a Kernel: KVConstAdd ..." << endl;
  #endif
  cl_k_mov = clCreateKernel(Program, "K_mov", &errCode);
  if (errCode != CL_SUCCESS) {
    cout << endl << "HOST-Error: Failed to create K_KVConstAdd" << endl << endl;
    return EXIT_FAILURE;
  }

  #ifdef ALL_MESSAGES
  cout << "HOST-Info: Creating a Kernel: KVConstAdd ..." << endl;
  #endif
  cl_k_apply_galois = clCreateKernel(Program, "K_apply_galois", &errCode);
  if (errCode != CL_SUCCESS) {
    cout << endl << "HOST-Error: Failed to create K_KVConstAdd" << endl << endl;
    return EXIT_FAILURE;
  }

  #ifdef ALL_MESSAGES
  cout << "HOST-Info: Creating a Kernel: KVConstAdd ..." << endl;
  #endif
  cl_k_ntt = clCreateKernel(Program, "K_ntt", &errCode);
  if (errCode != CL_SUCCESS) {
    cout << endl << "HOST-Error: Failed to create K_ntt" << endl << endl;
    return EXIT_FAILURE;
  }

  #ifdef ALL_MESSAGES
  cout << "HOST-Info: Creating a Kernel: KVConstAdd ..." << endl;
  #endif
  cl_k_scale = clCreateKernel(Program, "K_scale", &errCode);
  if (errCode != CL_SUCCESS) {
    cout << endl << "HOST-Error: Failed to create K_ntt" << endl << endl;
    return EXIT_FAILURE;
  }

  #ifdef ALL_MESSAGES
  cout << "HOST-Info: Creating a Kernel: KVConstAdd ..." << endl;
  #endif
  cl_k_lift = clCreateKernel(Program, "K_lift", &errCode);
  if (errCode != CL_SUCCESS) {
    cout << endl << "HOST-Error: Failed to create K_ntt" << endl << endl;
    return EXIT_FAILURE;
  }

  #ifdef ALL_MESSAGES
  cout << "HOST-Info: Creating a Kernel: KVConstAdd ..." << endl;
  #endif
  cl_k_mulInv = clCreateKernel(Program, "K_mulInv", &errCode);
  if (errCode != CL_SUCCESS) {
    cout << endl << "HOST-Error: Failed to create K_ntt" << endl << endl;
    return EXIT_FAILURE;
  }


  // ================================================================
  // Step 4: Prepare Data to Run Kernel
  // ================================================================
  //   o) Generate data for DataIn_1 array
  //   o) Generate data for DataIn_2 array
  //   o) Generate data for DataIn_3 array
  //   o) Allocate Memory to store the results: RES array
  //   o) Create Buffers in Global Memory to store data
  // ================================================================
  
  #ifdef ALL_MESSAGES
  cout << endl;
  cout << "HOST-Info: ============================================================= " << endl;
  cout << "HOST-Info: (Step 4) Prepare Data to Run Kernels                           " << endl;
  cout << "HOST-Info: ============================================================= " << endl;
  #endif

  // ADD //
  // dense100, mask, dense10, bias100

  void *ptr = nullptr;

  int *dense100_buf[13][4];
  int *mask_buf[4];
  int *dense10_buf[4];
  int *bias100_buf[4];


  for (int i = 0; i < 13; ++i) {
    for (int j = 0; j < 4; ++j) {    
      errCode = posix_memalign(&ptr, 4096, UNIT_INT_COUNT * sizeof(int)); checkError(errCode, __LINE__);
      dense100_buf[i][j] = reinterpret_cast<int*>(ptr);
    }
  }

  for (int i = 0; i < 4; ++i) {
    errCode = posix_memalign(&ptr, 4096, UNIT_INT_COUNT * sizeof(int)); checkError(errCode, __LINE__);
    mask_buf[i] = reinterpret_cast<int*>(ptr);
  }

  for (int i = 0; i < 4; ++i) {
    errCode = posix_memalign(&ptr, 4096, UNIT_INT_COUNT * sizeof(int)); checkError(errCode, __LINE__);
    dense10_buf[i] = reinterpret_cast<int*>(ptr);
  }

  for (int i = 0; i < 4; ++i) {
    errCode = posix_memalign(&ptr, 4096, UNIT_INT_COUNT * sizeof(int)); checkError(errCode, __LINE__);
    bias100_buf[i] = reinterpret_cast<int*>(ptr);
  }


  for (int i = 0; i < 13; ++i) {
    for (int j = 0; j < 4; ++j) {    
      for (int e = 0; e < UNIT_INT_COUNT; ++e) {
        dense100_buf[i][j][e] = dense100[i][j][e/8192][e%8192];
      }
    }
  }

  for (int i = 0; i < 4; ++i) {
    for (int e = 0; e < UNIT_INT_COUNT; ++e) {
      mask_buf[i][e] = mask[i][e/8192][e%8192];
    }
  }

  for (int i = 0; i < 4; ++i) {
    for (int e = 0; e < UNIT_INT_COUNT; ++e) {
      dense10_buf[i][e] = dense10[i][e/8192][e%8192];
    }
  }

  for (int i = 0; i < 4; ++i) {
    for (int e = 0; e < UNIT_INT_COUNT; ++e) {
      bias100_buf[i][e] = bias100[i][e/8192][e%8192];
    }
  }

  // -- 

  for (int i = 0; i < 13; ++i) {
    for (int j = 0; j < 4; ++j) {
      GM_dense100[i][j] = clCreateBuffer(Context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, UNIT_INT_COUNT * sizeof(int), 
       dense100_buf[i][j], &errCode);
      checkError(errCode, __LINE__);
    }
  }

  for (int i = 0; i < 4; ++i) {
    GM_mask[i] = clCreateBuffer(Context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, UNIT_INT_COUNT * sizeof(int), 
      mask_buf[i], &errCode);
    checkError(errCode, __LINE__);
  }

  for (int i = 0; i < 4; ++i) {
    GM_dense10[i] = clCreateBuffer(Context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, UNIT_INT_COUNT * sizeof(int), 
     dense10_buf[i], &errCode);
    checkError(errCode, __LINE__);
  }

  for (int i = 0; i < 4; ++i) {
    GM_bias100[i] = clCreateBuffer(Context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, UNIT_INT_COUNT * sizeof(int), 
      bias100_buf[i], &errCode);
    checkError(errCode, __LINE__);
  }

  // --

  for (int i = 0; i < 13; ++i)
    for (int j = 0; j < 4; ++j) {
      errCode = clEnqueueMigrateMemObjects(Command_Queue, 1, &GM_dense100[i][j], 0, 0, NULL, NULL);
      checkError(errCode, __LINE__);
    }

    for (int j = 0; j < 4; ++j) {
      errCode = clEnqueueMigrateMemObjects(Command_Queue, 1, &GM_mask[j], 0, 0, NULL, NULL);
      checkError(errCode, __LINE__);
    }

    for (int j = 0; j < 4; ++j) {
      errCode = clEnqueueMigrateMemObjects(Command_Queue, 1, &GM_dense10[j], 0, 0, NULL, NULL);
      checkError(errCode, __LINE__);
    }

    for (int j = 0; j < 4; ++j) {
      errCode = clEnqueueMigrateMemObjects(Command_Queue, 1, &GM_bias100[j], 0, 0, NULL, NULL);
      checkError(errCode, __LINE__);
    }

  clEnqueueBarrier(Command_Queue);

  //rlk, ct, gk

  int *rlk_buf[4][7][2];
  int *ct_buf[25][4][2];
  int *gk_buf[4][25][7][2];


  for (int i = 0; i < 4; ++i) {
    for (int j = 0; j < 7; ++j) {
      for (int k = 0; k < 2; ++k) {
        errCode = posix_memalign(&ptr, 4096, UNIT_INT_COUNT * sizeof(int)); checkError(errCode, __LINE__);
        rlk_buf[i][j][k] = reinterpret_cast<int*>(ptr);
      }
    }
  }

  for (int i = 0; i < 25; ++i) {
    for (int j = 0; j < 4; ++j) {
      for (int k = 0; k < 2; ++k) {
        errCode = posix_memalign(&ptr, 4096, UNIT_INT_COUNT * sizeof(int)); checkError(errCode, __LINE__);
        ct_buf[i][j][k] = reinterpret_cast<int*>(ptr);
      }
    }
  }

  for (int i = 0; i < 4; ++i) {
    for (int j = 0; j < 25; ++j) {
      for (int k = 0; k < 7; ++k) {
        for (int l = 0; l < 2; ++l) {
          errCode = posix_memalign(&ptr, 4096, UNIT_INT_COUNT * sizeof(int)); checkError(errCode, __LINE__);
          gk_buf[i][j][k][l] = reinterpret_cast<int*>(ptr);
        }
      }
    }
  }


  // --

  for (int i = 0; i < 4; ++i) {
    for (int j = 0; j < 7; ++j) {
      for (int k = 0; k < 2; ++k) {
        GM_rlk[i][j][k] = clCreateBuffer(Context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, UNIT_INT_COUNT * sizeof(int),
                                         rlk_buf[i][j][k], &errCode);

        checkError(errCode, __LINE__);
      }
    }
  }

  for (int i = 0; i < 25; ++i) {
    for (int j = 0; j < 4; ++j) {
      for (int k = 0; k < 2; ++k) {
        GM_ct[i][j][k] = clCreateBuffer(Context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, UNIT_INT_COUNT * sizeof(int),
                                        ct_buf[i][j][k], &errCode);
        checkError(errCode, __LINE__);
      }
    }
  }

  for (int i = 0; i < 4; ++i) {
    for (int j = 0; j < 25; ++j) {
      for (int k = 0; k < 7; ++k) {
        for (int l = 0; l < 2; ++l) {
          GM_gk[i][j][k][l] = clCreateBuffer(Context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, UNIT_INT_COUNT * sizeof(int),
                                             gk_buf[i][j][k][l], &errCode);
          checkError(errCode, __LINE__);
        }
      }
    }
  }


  // -- 


  for (int i = 0; i < 2; ++i) {
    temp1[i]  = clCreateBuffer(Context, CL_MEM_READ_WRITE, UNIT_INT_COUNT * sizeof(int), NULL, &errCode);
    checkError(errCode, __LINE__);
  }

  for (int i = 0; i < 2; ++i) {
    temp2[i]  = clCreateBuffer(Context, CL_MEM_READ_WRITE, UNIT_INT_COUNT * sizeof(int), NULL, &errCode);
    checkError(errCode, __LINE__);
  }

  for (int i = 0; i < 3; ++i) {
    temp3[i]  = clCreateBuffer(Context, CL_MEM_READ_WRITE, UNIT_INT_COUNT * sizeof(int), NULL, &errCode);
    checkError(errCode, __LINE__);
  }

  for (int i = 0; i < 3; ++i) {
    temp4[i]  = clCreateBuffer(Context, CL_MEM_READ_WRITE, UNIT_INT_COUNT * sizeof(int), NULL, &errCode);
    checkError(errCode, __LINE__);
  }

  for (int i = 0; i < 2; ++i) {
    temp5[i]  = clCreateBuffer(Context, CL_MEM_READ_WRITE, UNIT_INT_COUNT * sizeof(int), NULL, &errCode);
    checkError(errCode, __LINE__);
  }

  for (int i = 0; i < 7; ++i) {
    temp6[i]  = clCreateBuffer(Context, CL_MEM_READ_WRITE, UNIT_INT_COUNT * sizeof(int), NULL, &errCode);
    checkError(errCode, __LINE__);
  }

  for (int i = 0; i < 7; ++i) {
    temp7[i]  = clCreateBuffer(Context, CL_MEM_READ_WRITE, UNIT_INT_COUNT * sizeof(int), NULL, &errCode);
    checkError(errCode, __LINE__);
  }

  for (int i = 0; i < 7; ++i) {
    temp8[i]  = clCreateBuffer(Context, CL_MEM_READ_WRITE, UNIT_INT_COUNT * sizeof(int), NULL, &errCode);
    checkError(errCode, __LINE__);
  }

  for (int i = 0; i < 2; ++i) {
    temp9[i]  = clCreateBuffer(Context, CL_MEM_READ_WRITE, UNIT_INT_COUNT * sizeof(int), NULL, &errCode);
    checkError(errCode, __LINE__);
  }

  for (int i = 0; i < 2; ++i) {
    temp10[i] = clCreateBuffer(Context, CL_MEM_READ_WRITE, UNIT_INT_COUNT * sizeof(int), NULL, &errCode);
    checkError(errCode, __LINE__);
  }

  for (int i = 0; i < 5 ; ++i)
  for (int j = 0; j < 25; ++j)
  for (int k = 0; k < 4 ; ++k)
  for (int l = 0; l < 2 ; ++l) {
    temp_result[i][j][k][l] = clCreateBuffer(Context, CL_MEM_READ_WRITE, UNIT_INT_COUNT * sizeof(int), NULL, &errCode);
    checkError(errCode, __LINE__);
  }

  clEnqueueBarrier(Command_Queue);



  /* Start Network Server */

  in_port_t port = 17777;
  sockpp::socket_initializer sockInit;
  sockpp::tcp_acceptor acc(port);


  const int client_rlk_int_count = 4  * 7  * 2     * (7 * 8192);
  const int client_ct_int_count  = 25 * 4  * 2     * (7 * 8192);
  const int client_gk_int_count  = 4  * 25 * 7 * 2 * (7 * 8192); 

  // [-------------------------------------] data
  //  ^       rlk      ^    ct    ^   gk  ^         

  int *client_data_buffer = new int[client_rlk_int_count + client_ct_int_count + client_gk_int_count];
  int *rlk_ptr = client_data_buffer;
  int *ct_ptr  = client_data_buffer + client_rlk_int_count; 
  int *gk_ptr  = client_data_buffer + client_rlk_int_count + client_ct_int_count;

  if (!acc) {
    cerr << "Error creating the acceptor: " << acc.last_error_str() << endl;
    return EXIT_FAILURE;
  }
  cout << "Awaiting connections on port " << port << "..." << endl;

  while (true) {
    sockpp::inet_address peer;
    sockpp::tcp_socket sock = acc.accept(&peer);
    cout << "Received a connection request from " << peer << endl;
    if (!sock) {
      cerr << "Error accepting incoming connection: " << acc.last_error_str() << endl;
    }
    else {
      /* 6.1 Successfully Connected. */
      /* 6.2 Receive Data from Client */
      ssize_t n_bytes;
      char buf[16 * 1024];
      char *it = (char *)client_data_buffer;

      ssize_t rcved_bytes = 0;
      ssize_t total_bytes = (client_rlk_int_count + client_ct_int_count + client_gk_int_count) * sizeof(int);

      cout << "Start receiving data " << endl;

      while (rcved_bytes < total_bytes){

        n_bytes = sock.read(buf, sizeof(buf));
        memcpy(it, buf, n_bytes);
        it = it + n_bytes;
        rcved_bytes += n_bytes;

        // cout << "Progress: " << rcved_bytes << "/" << total_bytes << endl;
      }

      cout << "End receiving data "  << endl;

      int *ptr_data;

      ptr_data = rlk_ptr;
      for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 7; ++j) {
          for (int k = 0; k < 2; ++k) {
            for (int e = 0; e < 7 * 8192; ++e){
              rlk_buf[i][j][k][e] = *(ptr_data++);
            }
            for (int e = 7 * 8192; e < 8 * 8192; ++e){
              rlk_buf[i][j][k][e] = 0;
            }
          }
        }
      }

      ptr_data = ct_ptr;
      for (int i = 0; i < 25; ++i) {
        for (int j = 0; j < 4; ++j) {
          for (int k = 0; k < 2; ++k) {
            for (int e = 0; e < 7 * 8192; ++e){
              ct_buf[i][j][k][e] = *(ptr_data++);
            }
            for (int e = 7 * 8192; e < 8 * 8192; ++e){
              ct_buf[i][j][k][e] = 0;
            }
          }
        }
      }

      ptr_data = gk_ptr;
      for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 25; ++j) {
          for (int k = 0; k < 7; ++k) {
            for (int l = 0; l < 2; ++l) {
              for (int e = 0; e < 7 * 8192; ++e){
                gk_buf[i][j][k][l][e] = *(ptr_data++);
              }
              for (int e = 7 * 8192; e < 8 * 8192; ++e){
                gk_buf[i][j][k][l][e] = 0;
              }
            }
          }
        }
      }

      /* 6.3 Send Data to Device Memory */

      cout << "Transfering data to device memory" << endl;

      for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 7; ++j) {
          for (int k = 0; k < 2; ++k) {
            errCode = clEnqueueMigrateMemObjects(Command_Queue, 1, &GM_rlk[i][j][k], 0, 0, NULL, NULL);
            checkError(errCode, __LINE__);
          }
        }
      }

      for (int i = 0; i < 25; ++i) {
        for (int j = 0; j < 4; ++j) {
          for (int k = 0; k < 2; ++k) {
            errCode = clEnqueueMigrateMemObjects(Command_Queue, 1, &GM_ct[i][j][k], 0, 0, NULL, NULL);
            checkError(errCode, __LINE__);
          }
        }
      }

      for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 25; ++j) {
          for (int k = 0; k < 7; ++k) {
            for (int l = 0; l < 2; ++l) {
              errCode = clEnqueueMigrateMemObjects(Command_Queue, 1, &GM_gk[i][j][k][l], 0, 0, NULL, NULL);
              checkError(errCode, __LINE__);
            }
          }
        }
      }

      clEnqueueBarrier(Command_Queue);

      /* 6.4 Processing */
      // TODO

      for (int i = 0; i < 8192; i++)
      		  cout << ct_buf[0][0][1][i+8192*7] << " ";
      	  cout << endl;
      cout << "Enter Inference ..." << endl;
      inference_test();
      clEnqueueBarrier(Command_Queue);
      cout << "Leave Interface ..." << endl;

      /* 6.5 Get Result and send it Back to Client */
      int res[4*2*7*8192];


	  for (int j = 0; j < 4; ++j) {
	    for (int k = 0; k < 2; ++k) {
		  errCode = clEnqueueMigrateMemObjects(Command_Queue, 1, &GM_ct[0][j][k], CL_MIGRATE_MEM_OBJECT_HOST, 0, NULL, NULL);
		  checkError(errCode, __LINE__);
		  clEnqueueBarrier(Command_Queue);
		  sock.write_n(ct_buf[0][j][k], 7*8192*sizeof(int));
	    }
	  }
//	  for (int i = 0; i < 8192; i++)
//		  cout << ct_buf[0][0][1][i+8192*7] << " ";
//	  cout << endl;




      cout << "Connection closed from " << sock.peer_address() << endl;
    }

    break; // only accept 1 request now
  }

  delete[] client_data_buffer;

  clEnqueueBarrier(Command_Queue);

  clReleaseDevice(Target_Device_ID); // Only available in OpenCL >= 1.2


  for (int i = 0; i < 13; ++i) {
    for (int j = 0; j < 4; ++j) {
      errCode = clReleaseMemObject(GM_dense100[i][j]);
      checkError(errCode, __LINE__);
    }
  }

  for (int i = 0; i < 4; ++i) {
    errCode = clReleaseMemObject(GM_mask[i]);
    checkError(errCode, __LINE__);
  }

  for (int i = 0; i < 4; ++i) {
    errCode = clReleaseMemObject(GM_dense10[i]);
    checkError(errCode, __LINE__);
  }

  for (int i = 0; i < 4; ++i) {
    errCode = clReleaseMemObject(GM_bias100[i]);
    checkError(errCode, __LINE__);
  }

  for (int i = 0; i < 4; ++i) {
    for (int j = 0; j < 7; ++j) {
      for (int k = 0; k < 2; ++k) {
        errCode = clReleaseMemObject(GM_rlk[i][j][k]);
        checkError(errCode, __LINE__);
      }
    }
  }

  for (int i = 0; i < 25; ++i) {
    for (int j = 0; j < 4; ++j) {
      for (int k = 0; k < 2; ++k) {
        errCode = clReleaseMemObject(GM_ct[i][j][k]);
        checkError(errCode, __LINE__);
      }
    }
  }

  for (int i = 0; i < 4; ++i) {
    for (int j = 0; j < 25; ++j) {
      for (int k = 0; k < 7; ++k) {
        for (int l = 0; l < 2; ++l) {
          errCode = clReleaseMemObject(GM_gk[i][j][k][l]);
          checkError(errCode, __LINE__);
        }
      }
    }
  }



  for (int i = 0; i < 2; ++i) {
    errCode = clReleaseMemObject(temp1[i]);
    checkError(errCode, __LINE__);
  }

  for (int i = 0; i < 2; ++i) {
    errCode = clReleaseMemObject(temp2[i]);
    checkError(errCode, __LINE__);
  }

  for (int i = 0; i < 3; ++i) {
    errCode = clReleaseMemObject(temp3[i]);
    checkError(errCode, __LINE__);
  }

  for (int i = 0; i < 3; ++i) {
    errCode = clReleaseMemObject(temp4[i]);
    checkError(errCode, __LINE__);
  }

  for (int i = 0; i < 2; ++i) {
    errCode = clReleaseMemObject(temp5[i]);
    checkError(errCode, __LINE__);
  }

  for (int i = 0; i < 7; ++i) {
    errCode = clReleaseMemObject(temp6[i]);
    checkError(errCode, __LINE__);
  }

  for (int i = 0; i < 7; ++i) {
    errCode = clReleaseMemObject(temp7[i]);
    checkError(errCode, __LINE__);
  }

  for (int i = 0; i < 7; ++i) {
    errCode = clReleaseMemObject(temp8[i]);
    checkError(errCode, __LINE__);
  }

  for (int i = 0; i < 2; ++i) {
    errCode = clReleaseMemObject(temp9[i]);
    checkError(errCode, __LINE__);
  }

  for (int i = 0; i < 2; ++i) {
    errCode = clReleaseMemObject(temp10[i]);
    checkError(errCode, __LINE__);
  }

  for (int i = 0; i < 5 ; ++i)
  for (int j = 0; j < 25; ++j)
  for (int k = 0; k < 4 ; ++k)
  for (int l = 0; l < 2 ; ++l) {
    errCode = clReleaseMemObject(temp_result[i][j][k][l]);
    checkError(errCode, __LINE__);
  }

  clReleaseKernel(cl_k_add);
  clReleaseKernel(cl_k_mulWise);
  clReleaseKernel(cl_k_mulConst);
  clReleaseKernel(cl_k_mov);
  clReleaseKernel(cl_k_apply_galois);
  clReleaseKernel(cl_k_ntt);
  clReleaseKernel(cl_k_scale);
  clReleaseKernel(cl_k_lift);
  clReleaseKernel(cl_k_mulInv);

  clReleaseProgram(Program);
  clReleaseCommandQueue(Command_Queue);
  clReleaseContext(Context);

  free(Platform_IDs);
  free(Device_IDs);

  for (int i = 0; i < 13; ++i) {
    for (int j = 0; j < 4; ++j) {    
      free(dense100_buf[i][j]);
    }
  }

  for (int i = 0; i < 4; ++i) {
    free(mask_buf[i]);
  }

  for (int i = 0; i < 4; ++i) {
    free(dense10_buf[i]);
  }

  for (int i = 0; i < 4; ++i) {
    free(bias100_buf[i]);
  }

  for (int i = 0; i < 4; ++i) {
    for (int j = 0; j < 7; ++j) {
      for (int k = 0; k < 2; ++k) {
        free(rlk_buf[i][j][k]);
      }
    }
  }

  for (int i = 0; i < 25; ++i) {
    for (int j = 0; j < 4; ++j) {
      for (int k = 0; k < 2; ++k) {
        free(ct_buf[i][j][k]);
      }
    }
  }

  for (int i = 0; i < 4; ++i) {
    for (int j = 0; j < 25; ++j) {
      for (int k = 0; k < 7; ++k) {
        for (int l = 0; l < 2; ++l) {
          free(gk_buf[i][j][k][l]);
        }
      }
    }
  }



  cout << endl << "HOST-Info: DONE" << endl << endl;

  return EXIT_SUCCESS;
}

