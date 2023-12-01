#include <cuda_runtime.h>
#include <cstdio>
#include <random>
#include <curand_kernel.h>
#include "cudaMatrix.cuh"


__global__ void matMulKernel(CudaMatrix *A, CudaMatrix *B, CudaMatrix *C);


__global__ void matAddKernel(CudaMatrix *A, CudaMatrix *B, CudaMatrix *C);

__global__ void matSubKernel(CudaMatrix *A, CudaMatrix *B, CudaMatrix *C);

__global__ void matScalarMultKernel(CudaMatrix *A, int k, CudaMatrix *B);

__global__ void hadamardProductKernel(CudaMatrix *A, CudaMatrix *B, CudaMatrix *C);


//Currently only support 2-D matrices
CudaMatrix * matMul(CudaMatrix *A, CudaMatrix *B);

CudaMatrix * matAdd(CudaMatrix *A, CudaMatrix *B);

CudaMatrix * matSub(CudaMatrix *A, CudaMatrix *B);

CudaMatrix * matScalarMult(CudaMatrix *A, int k);

CudaMatrix * haramardProd(CudaMatrix *A, CudaMatrix *B);
