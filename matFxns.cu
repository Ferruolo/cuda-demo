
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


__global__ void matAddKernel(CudaMatrix *A, CudaMatrix *B, CudaMatrix *C) {
    int coords[] = {static_cast<int>(threadIdx.x), static_cast<int>(threadIdx.y)};
    getElement(C, coords) = getElement(A, coords) + getElement(B, coords);
}

__global__ void matSubKernel(CudaMatrix *A, CudaMatrix *B, CudaMatrix *C) {
    int coords[] = {static_cast<int>(threadIdx.x), static_cast<int>(threadIdx.y)};
    getElement(C, coords) = getElement(A, coords) - getElement(B, coords);
}

__global__ void matScalarMultKernel(CudaMatrix *A, int k, CudaMatrix *B) {
    int coords[] = {static_cast<int>(threadIdx.x), static_cast<int>(threadIdx.y)};
    getElement(B, coords) = getElement(A, coords) * k;
}

__global__ void hadamardProductKernel(CudaMatrix *A, CudaMatrix *B, CudaMatrix *C) {
    int coords[] = {static_cast<int>(threadIdx.x), static_cast<int>(threadIdx.y)};
    getElement(C, coords) = getElement(A, coords) * getElement(B, coords);
}


//Currently only support 2-D matrices
CudaMatrix * matMul(CudaMatrix *A, CudaMatrix *B){
    if (getShape(A)[1] != getShape(B)[0]) {
        printf("Matrix Shapes not Compatible\n");
        return nullptr;
    }


    int newMatShape[] = {getShape(A)[0], getShape(B)[1]};
    dim3 newShape(newMatShape[0], newMatShape[1]);
    int middle_dim = getShape(A)[1];

    CudaMatrix *C = BuildMatrixZeros(newMatShape, 2);

    CudaMatrix *A_d, *B_d, *C_d;
    cudaMalloc(&A_d, sizeof(CudaMatrix));
    cudaMalloc(&B_d, sizeof(CudaMatrix));
    cudaMalloc(&C_d, sizeof(CudaMatrix));

    cudaMemcpy(A_d, A, sizeof(CudaMatrix), cudaMemcpyHostToDevice);
    cudaMemcpy(B_d, B, sizeof(CudaMatrix), cudaMemcpyHostToDevice);
    cudaMemcpy(C_d, C, sizeof(CudaMatrix), cudaMemcpyHostToDevice);

    matMulKernel<<<middle_dim, newShape>>>(A_d, B_d, C_d);

    cudaDeviceSynchronize();
    cudaMemcpy(C, C_d, sizeof(CudaMatrix), cudaMemcpyDeviceToHost);

    return C;
}

CudaMatrix * matAdd(CudaMatrix *A, CudaMatrix *B) {
    const int *aShape = getShape(A);
    const int *bShape = getShape(B);

    for (int i = 0; i < A->numDim; ++i){
        if (aShape[i] != bShape[i]){

            printf("Matrix Shapes not Compatible: %d != %d \n", aShape[i], bShape[i]);
            return nullptr;
        }
    }
    dim3 newShape(aShape[0], aShape[1]);
    CudaMatrix *C = BuildMatrixZeros(aShape, A->numDim);


    CudaMatrix *A_d, *B_d, *C_d;
    cudaMalloc(&A_d, sizeof(CudaMatrix));
    cudaMalloc(&B_d, sizeof(CudaMatrix));
    cudaMalloc(&C_d, sizeof(CudaMatrix));

    cudaMemcpy(A_d, A, sizeof(CudaMatrix), cudaMemcpyHostToDevice);
    cudaMemcpy(B_d, B, sizeof(CudaMatrix), cudaMemcpyHostToDevice);
    cudaMemcpy(C_d, C, sizeof(CudaMatrix), cudaMemcpyHostToDevice);
    matAddKernel<<<1, newShape>>>(A_d, B_d, C_d);

    cudaDeviceSynchronize();
    cudaMemcpy(C, C_d, sizeof(CudaMatrix), cudaMemcpyDeviceToHost);

    return C;
}



CudaMatrix * matSub(CudaMatrix *A, CudaMatrix *B) {
    const int *aShape = getShape(A);
    const int *bShape = getShape(B);

    for (int i = 0; i < A->numDim; ++i){
        if (aShape[i] != bShape[i]){

            printf("Matrix Shapes not Compatible: %d != %d \n", aShape[i], bShape[i]);
            return nullptr;
        }
    }
    dim3 newShape(aShape[0], aShape[1]);
    CudaMatrix *C = BuildMatrixZeros(aShape, A->numDim);


    CudaMatrix *A_d, *B_d, *C_d;
    cudaMalloc(&A_d, sizeof(CudaMatrix));
    cudaMalloc(&B_d, sizeof(CudaMatrix));
    cudaMalloc(&C_d, sizeof(CudaMatrix));

    cudaMemcpy(A_d, A, sizeof(CudaMatrix), cudaMemcpyHostToDevice);
    cudaMemcpy(B_d, B, sizeof(CudaMatrix), cudaMemcpyHostToDevice);
    cudaMemcpy(C_d, C, sizeof(CudaMatrix), cudaMemcpyHostToDevice);
    matSubKernel<<<1, newShape>>>(A_d, B_d, C_d);

    cudaDeviceSynchronize();
    cudaMemcpy(C, C_d, sizeof(CudaMatrix), cudaMemcpyDeviceToHost);

    return C;
}


CudaMatrix * matScalarMult(CudaMatrix *A, int k) {
    const int *aShape = getShape(A);


    dim3 newShape(aShape[0], aShape[1]);
    CudaMatrix *C = BuildMatrixZeros(aShape, A->numDim);


    CudaMatrix *A_d, *C_d;
    cudaMalloc(&A_d, sizeof(CudaMatrix));

    cudaMalloc(&C_d, sizeof(CudaMatrix));

    cudaMemcpy(A_d, A, sizeof(CudaMatrix), cudaMemcpyHostToDevice);

    cudaMemcpy(C_d, C, sizeof(CudaMatrix), cudaMemcpyHostToDevice);
    matScalarMultKernel<<<1, newShape>>>(A_d, k, C_d);

    cudaDeviceSynchronize();
    cudaMemcpy(C, C_d, sizeof(CudaMatrix), cudaMemcpyDeviceToHost);

    return C;
}

CudaMatrix * haramardProd(CudaMatrix *A, CudaMatrix *B) {
    const int *aShape = getShape(A);
    const int *bShape = getShape(B);

    for (int i = 0; i < A->numDim; ++i){
        if (aShape[i] != bShape[i]){

            printf("Matrix Shapes not Compatible: %d != %d \n", aShape[i], bShape[i]);
            return nullptr;
        }
    }
    dim3 newShape(aShape[0], aShape[1]);
    CudaMatrix *C = BuildMatrixZeros(aShape, A->numDim);


    CudaMatrix *A_d, *B_d, *C_d;
    cudaMalloc(&A_d, sizeof(CudaMatrix));
    cudaMalloc(&B_d, sizeof(CudaMatrix));
    cudaMalloc(&C_d, sizeof(CudaMatrix));

    cudaMemcpy(A_d, A, sizeof(CudaMatrix), cudaMemcpyHostToDevice);
    cudaMemcpy(B_d, B, sizeof(CudaMatrix), cudaMemcpyHostToDevice);
    cudaMemcpy(C_d, C, sizeof(CudaMatrix), cudaMemcpyHostToDevice);
    hadamardProductKernel<<<1, newShape>>>(A_d, B_d, C_d);

    cudaDeviceSynchronize();
    cudaMemcpy(C, C_d, sizeof(CudaMatrix), cudaMemcpyDeviceToHost);

    return C;
}

