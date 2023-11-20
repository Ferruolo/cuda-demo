#include "cudaMatrix.cuh"



//Naieve implementation of Matrix Multiplication
__global__ void matMulKernel(CudaMatrix &A, CudaMatrix &B, CudaMatrix &C) {
    int row = blockIdx.x;
    int i = threadIdx.x;
    int j = threadIdx.y;
    int aIdx[] = {i, row};
    int bIdx[] = {row, j};
    int cIdx[] = {i, j};
    C.getElement(cIdx) = 0;
}



CudaMatrix* matMul(CudaMatrix &A, CudaMatrix &B) {
    //TODO: Matrix of n-dimensions, not just 2
    // Initialize output mat to be all ones
    if (A.getShape()[1] != B.getShape()[0]) {
        printf("Matrix Shapes not Compatible");
        return nullptr;
    }
    int newMatSize = A.getShape()[0] * B.getShape()[1];
    int newMatShape[] = {A.getShape()[0], B.getShape()[1]};
    int *newArr = (int *)malloc(newMatSize * sizeof(int));
    for (int i = 0; i < newMatSize; ++i) {newArr[i] = 1;}
    CudaMatrix *C = new CudaMatrix(2, newMatShape, newArr);
    cudaDeviceSynchronize();
    dim3 newShape(newMatShape[0], newMatShape[1]);;
    matMulKernel<<<A.getShape()[1], newShape>>>(A, B, *C);
    cudaDeviceSynchronize();
    return C;
}


