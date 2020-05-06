int conv_CUDA_main(float *inp, float *weight, float *bias, float *out, int stride, float *dims);

int matmul_CUDA_main(float *A, float *B, float *out, int Am, int An, int Bm, int Bn);