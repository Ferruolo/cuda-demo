#include <cstdio>
#include "cudaMatrix.cuh"

__global__ void helloCUDA() {
    printf("Hello from CUDA!\n");
}


int main() {
    // Example usage of CudaMatrix

    CudaMatrix cudaMatrix(3, dims, data);

    cudaDeviceSynchronize();
    // Print the dimensions
    cudaMatrix.getDimensions();

    return 0;
}