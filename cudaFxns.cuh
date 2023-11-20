#include "cudaMatrix.cuh"



//Naieve implementation of Matrix Multiplication
__global__ void matMulKernel(CudaMatrix &A, CudaMatrix &B, CudaMatrix &C) {
    int row = blockIdx.x;
    int i = threadIdx.x;
    int j = threadIdx.y;
    int aIdx[] = {i, row};
    int bIdx[] = {row, j};

    int cIdx[] = {i, j};

    C.getElement(cIdx) = A.getElement(aIdx) + B.getElement(bIdx);
}



void matMul(CudaMatrix &A, CudaMatrix&B, CudaMatrix &C) {
    // TODO: Redefine C here so that it can take product of A and B
    // For now we assume everything is a 5x5 matrix, cause I want to be asleep by 2
    if (A.getShape()[1] == B.getShape()[0]){
        dim3 threadsPerBlock(A.getShape()[0], B.getShape()[1]);
        matMulKernel<<<A.getShape()[1], threadsPerBlock>>>(A, B, C);
    }
}

__global__ void cuRandArr(int *randArray) {
    int tid = threadIdx.x;
    curandState state;
    curand_init(clock64(), tid, 0, &state);
    int r = (int)(curand_uniform(&state) * 20);
    randArray[tid] = (r);
}


