#include "cudaMatrix.cuh"


__device__ void MatMul(CudaMatrix &A, CudaMatrix &B, CudaMatrix &C) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    C[i] = A[i] + B[i];

}

