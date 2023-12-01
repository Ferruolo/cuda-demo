#include <cuda_runtime.h>
#include <cstdio>
#include <random>
#include <curand_kernel.h>


__global__ void cuRandArrInit(int *randArray);

__global__ void cuConstArrInit(int *randArray, const int c);

// TODO: add support for doubles, and possibly matrices with more support
struct CudaMatrix {
    int * mat = nullptr;
    int * shape = nullptr;
    int size = 1;
    int numDim = 0;
    int *elementsPerDim = nullptr;
//    ~CudaMatrix(){
//        cudaFree(&mat);
//        cudaFree(&shape);
//        cudaFree(&elementsPerDim);
//    }
};


CudaMatrix *createGenMatrix(const int *dims, int numDimensions);

//Randomly initialized matrix
CudaMatrix* BuildRandomMatrix (int dims[], int numDimensions);


CudaMatrix* BuildMatrixConst (const int dims[], int numDimensions, const int &c);

CudaMatrix* BuildMatrixZeros (const int dims[], int numDimensions);

void printMatrix(const CudaMatrix *mat);


const int * getShape(CudaMatrix *mat);

__device__ void setElement(CudaMatrix *m, const int*coords, const int &c);


__device__ int &getElement(CudaMatrix *m, const int*coords);
