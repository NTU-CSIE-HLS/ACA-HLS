int power(long long x, unsigned int y, int p);

void keyswitching(cl_mem ct2,
                  cl_mem k[7][2],
                  cl_mem out[2]);

// HE primitives
void MulConst(cl_mem in[4][2],
              cl_mem out[4][2],
              int constant);

void MulWise(cl_mem in1[4][2],
             cl_mem in2[4],
             cl_mem out[4][2]);
void HMul(cl_mem in1[4][2],
          cl_mem in2[4][2],
          cl_mem out[4][2]);
void HAdd(cl_mem in1[4][2],
          cl_mem in2[4][2],
          cl_mem out[4][2]);
void rotate_row(cl_mem in1[4][2],
                int r,
                cl_mem out[4][2]);
void rotate_column(cl_mem in1[4][2],
                   int r, // +- 1, 2, 4, ...
                   cl_mem out[4][2]);
void CNN(cl_mem in1[25], cl_mem out);
void Square(cl_mem in1[2]);
void inference();
void inference_test();
