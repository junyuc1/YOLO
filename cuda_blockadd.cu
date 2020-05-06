#include "cuda.h"
#include "stdio.h"
#include "stdlib.h"
#define LBLK 8

__device__ int get_index(int N, int C, int H, int W, int a, int b, int c, int d) {
    return a * C * H * W + b * W * H + c * W + d;
}

__device__ int RM(int i, int j, int N) {
    return i * N + j;
}

__global__ void conv_kernel(float *weight, float *inp, float *bias, float *out, int N, int C, int H, int W, int num_filter, int filter_depth, int filter_H, int filter_W, int out_size, int stride) {
    int h = blockIdx.y * blockDim.y + threadIdx.y;
    int w = blockIdx.x * blockDim.x + threadIdx.x;
    int k_d = blockIdx.z * blockDim.z + threadIdx.z;
    int n, c, k_h, k_w;
    if (h < out_size && w < out_size && k_d < num_filter) {
        float my_bias = bias[k_d];
        int kernel_x = w * stride;
        int kernel_y = h * stride;
        for (n = 0; n < N; n ++) {
            float cur_val = my_bias;
            for (c = 0; c < filter_depth; c ++) { 
                for (k_h = 0; k_h < filter_H; k_h ++) {
                    for (k_w = 0; k_w < filter_W; k_w ++) {
                        float target_weight = weight[get_index(num_filter, filter_depth, filter_H, filter_W, k_d, c, k_h, k_w)];
                        float target_inp = inp[get_index(N, C, H, W, n, c, kernel_y + k_h, kernel_x + k_w)];
                        cur_val += target_inp * target_weight;
                    }
                }
            }
            out[get_index(N, num_filter, out_size, out_size, n, k_d, h, w)] = cur_val; 
        }
    }
}

__global__ void matmul_kernel(float *A, float *B, float *out, int Am, int An, int Bm, int Bn) {
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int bi = threadIdx.y;
    int bj = threadIdx.x;
    __shared__ float subA[LBLK * LBLK];
    __shared__ float subB[LBLK * LBLK];

    float sum = 0.0;
    int k;

    for (k = 0; k < An; k += LBLK) {
        if (i < Am && k + bj < An) {
            subA[RM(bi, bj, LBLK)] = A[RM(i, k + bj, An)];
        } else {
            subA[RM(bi, bj, LBLK)] = 0.0;
        }
        if (k + bi < Bm && j < Bn) {
            subB[RM(bi, bj, LBLK)] = B[RM(k + bi, j, Bn)];
        } else {
            subB[RM(bi, bj, LBLK)] = 0.0;
        }
        __syncthreads();
        
        for (int bk = 0; bk < LBLK; bk++) {
            sum += subA[RM(bi, bk, LBLK)] * subB[RM(bk, bj, LBLK)];
        }
        __syncthreads();
    }

    if (i < Am && j < Bn) {
        out[RM(i, j, Bn)] = sum;
    }
    
}

extern "C" int conv_CUDA_main(float *inp, float *weight, float *bias, float *out, int stride, float *dims) {
    int N = (int)dims[0];
    int C = (int)dims[1];
    int H = (int)dims[2];
    int W = (int)dims[3];
    int num_filter = (int)dims[4];
    int filter_depth = (int)dims[5];
    int filter_H = (int)dims[6];
    int filter_W = (int)dims[7];
    int out_size = (int)dims[8];

    float *dev_weight, *dev_inp, *dev_bias, *dev_out;

    cudaMalloc(&dev_weight, num_filter * filter_depth * filter_H * filter_W * sizeof(float));
    cudaMalloc(&dev_inp, N * C * H * W * sizeof(float));
    cudaMalloc(&dev_bias, num_filter * sizeof(float));
    cudaMalloc(&dev_out, N * num_filter * out_size * out_size * sizeof(float));

    cudaMemcpy(dev_weight, weight, num_filter * filter_depth * filter_H * filter_W * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_inp, inp, N * C * H * W * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_bias, bias, num_filter * sizeof(float), cudaMemcpyHostToDevice);

    dim3 blockDim(8, 8, 8);
    dim3 gridDim((out_size + blockDim.x - 1) / blockDim.x, (out_size + blockDim.y - 1) / blockDim.y, (num_filter + blockDim.z - 1) / blockDim.z);

    conv_kernel<<<gridDim, blockDim>>>(dev_weight, dev_inp, dev_bias, dev_out, N, C, H, W, num_filter, filter_depth, filter_H, filter_W, out_size, stride);
    
    cudaMemcpy(out, dev_out, N * num_filter * out_size * out_size * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(dev_weight);
    cudaFree(dev_inp);
    cudaFree(dev_bias);
    cudaFree(dev_out);

    return 0;
}

extern "C" int matmul_CUDA_main(float *A, float *B, float *out, int Am, int An, int Bm, int Bn) {
    float *dev_A, *dev_B, *dev_bias, *dev_out;   

    cudaMalloc(&dev_A, Am * An * sizeof(float));
    cudaMalloc(&dev_B, Bm * Bn * sizeof(float));
    cudaMalloc(&dev_out, Am * Bn * sizeof(float));

    cudaMemcpy(dev_A, A, Am * An * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_B, B, Bm * Bn * sizeof(float), cudaMemcpyHostToDevice);

    dim3 blockDim(8, 8);
    dim3 gridDim((Bn + blockDim.x - 1) / blockDim.x, (Am + blockDim.y - 1) / blockDim.y);

    matmul_kernel<<<gridDim, blockDim>>>(dev_A, dev_B, dev_out, Am, An, Bm, Bn);
    
    cudaMemcpy(out, dev_out, Am * Bn * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(dev_A);
    cudaFree(dev_B);
    cudaFree(dev_out);

    return 0;
}