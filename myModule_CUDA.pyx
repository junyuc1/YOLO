import numpy as np
cimport numpy as np

cdef extern from './cuda_blockadd.h':
    int conv_CUDA_main(float *inp, float *weight, float *bias, float *out, int stride, float *dims) nogil
    int matmul_CUDA_main(float *A, float *B, float *out, int Am, int An, int Bm, int Bn) nogil

def Conv_Par_CUDA(np.ndarray[np.float32_t, ndim=4] inp,
        np.ndarray[np.float32_t, ndim=4] weight,
        np.ndarray[np.float32_t, ndim=1] bias,
        np.ndarray[np.float32_t, ndim=1] dims,
        stride):
    cdef np.ndarray[np.float32_t, ndim=4] out
    
    N = inp.shape[0]
    C = inp.shape[1]
    H = inp.shape[2]
    W = inp.shape[3]

    num_filter = weight.shape[0]
    filter_depth = weight.shape[1]
    filter_H = weight.shape[2]
    filter_W = weight.shape[3]

    out_size = (W - filter_W) // stride + 1

    out = np.zeros((N, num_filter, out_size, out_size), dtype="float32")

    conv_CUDA_main(&inp[0, 0, 0, 0], &weight[0, 0, 0, 0], &bias[0], &out[0, 0, 0, 0], stride, &dims[0])
    
    return out

def Matmul_Par_CUDA(np.ndarray[np.float32_t, ndim=2] A,
        np.ndarray[np.float32_t, ndim=2] B,
        Am, An, Bm, Bn):
    cdef np.ndarray[np.float32_t, ndim=2] out

    out = np.zeros((Am, Bn), dtype="float32")

    matmul_CUDA_main(&A[0, 0], &B[0, 0], &out[0, 0], Am, An, Bm, Bn)
    
    return out