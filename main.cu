#include <cstdio>
#include "matFxns.cuh"

__global__ void helloCUDA() {
    printf("Hello from CUDA!\n");
}

int main() {
    // Example usage of CudaMatrix
    int num_dim = 2;
    int d1[] = {5, 5};
    int d2[] = {5, 5};
    auto *A = BuildRandomMatrix(d1, num_dim);
    auto *B = BuildRandomMatrix(d2, num_dim);
    auto *ones = BuildMatrixConst(d1, num_dim, 1);

    printMatrix(A);
    printMatrix(B);
    auto *C = matMul(A, B);
    printMatrix(C);



    auto *D = matAdd(A, ones);
    printf("Arrived\n");
    printMatrix(D);
    return 0;
}
