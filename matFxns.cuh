#include <cuda_runtime.h>
#include <cstdio>
#include <random>
#include <curand_kernel.h>
#include "cudaMatrix.cuh"


__global__ void matMulKernel(CudaMatrix *A, CudaMatrix *B, CudaMatrix *C) {
    int row = blockIdx.x;
    int i = threadIdx.x;
    int j = threadIdx.y;
    int aIdx[] = {i, row};
    int bIdx[] = {row, j};
    int cIdx[] = {i, j};

    int a = getElement(A, aIdx);
    int b = getElement(B, bIdx);
    int &c = getElement(C, cIdx);
    atomicAdd(&c, a*b);
}


//Currently only support 2-D matrices
CudaMatrix * matMul(CudaMatrix *A, CudaMatrix *B){
    if (getShape(A)[1] != getShape(B)[0]) {
        printf("Matrix Shapes not Compatible");
        return nullptr;
    }


    CudaMatrix *A_d, *B_d, *C_d;
    cudaMalloc(&A_d, sizeof(CudaMatrix));
    cudaMalloc(&B_d, sizeof(CudaMatrix));

    cudaMemcpy(A_d, A, sizeof(CudaMatrix), cudaMemcpyHostToDevice);
    cudaMemcpy(B_d, B, sizeof(CudaMatrix), cudaMemcpyHostToDevice);


    int newMatShape[] = {getShape(A)[0], getShape(B)[1]};
    dim3 newShape(newMatShape[0], newMatShape[1]);;
    int middle_dim = getShape(A)[1];

    CudaMatrix *C = BuildMatrixZeros(newMatShape, 2);
    cudaMalloc(&C_d, sizeof(CudaMatrix));
    cudaMemcpy(C_d, C, sizeof(CudaMatrix), cudaMemcpyHostToDevice);

    matMulKernel<<<middle_dim, newShape>>>(A_d, B_d, C_d);

    cudaDeviceSynchronize();
    cudaMemcpy(C, C_d, sizeof(CudaMatrix), cudaMemcpyDeviceToHost);

    cudaFree(A_d);
    cudaFree(B_d);
    cudaFree(C_d);
    return C;
}