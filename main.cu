#include <cstdio>
#include "cudaFxns.cuh"

__global__ void helloCUDA() {
    printf("Hello from CUDA!\n");
}


int main() {
    // Example usage of CudaMatrix
    int num_dim = 2;
    int dimensions[] = {5, 5};

    CudaMatrix A(num_dim, dimensions);

    CudaMatrix B(num_dim, dimensions);

    CudaMatrix C(num_dim, dimensions);
    cudaDeviceSynchronize();
    A.printMatrix();
    printf("------------------\n");
    B.printMatrix();
    printf("------------------\n");
//    C.printMatrix();
//    printf("------------------\n");
    matMul(A, B, C);
    cudaDeviceSynchronize();
    // Print the dimensions
    C.printMatrix();

    return 0;
}


