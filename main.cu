#include <cstdio>
#include "matFxns.cuh"

__global__ void helloCUDA() {
    printf("Hello from CUDA!\n");
}


int main() {
    // Example usage of CudaMatrix
    int num_dim = 2;
    int dimensions[] = {5, 5};

    CudaMatrix A(num_dim, dimensions);

    CudaMatrix B(num_dim, dimensions);

    cudaDeviceSynchronize();
    A.printMatrix();
    printf("------------------\n");
    B.printMatrix();
    printf("------------------\n");
//    for (int i = 0; i < 5; ++i){
//        for (int j = 0; j < 5; ++j){
//            int loc[] = {i, j};
//            printf("%d ", A.getElement(loc));
//        }
//        printf("\n");
//    }


    CudaMatrix * C = matMul(A, B);
//     Print the dimensions
    if (C) C->printMatrix();
    delete C;

    return 0;
}


