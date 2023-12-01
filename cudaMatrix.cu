#include "cudaMatrix.cuh"

__global__ void cuRandArrInit(int *randArray) {
    int tid = threadIdx.x;
    curandState state;
    curand_init(clock64(), tid, 0, &state);
    int r = (int)(curand_uniform(&state) * 20);
    randArray[tid] = (r);
}

__global__ void cuConstArrInit(int *randArray, const int c) {
    int tid = threadIdx.x;
    randArray[tid] = (c);
}


CudaMatrix *createGenMatrix(const int *dims, int numDimensions) {// TODO: add cuda feature

    //Create Matrix
    auto *mat = new CudaMatrix;

    cudaMallocManaged(&(mat->elementsPerDim), numDimensions * sizeof(int));


    // Initialize Size and elements per dim
    for(int i = 0; i < numDimensions; ++i){
        if (i > 0) {
            for (int j = 0; j < i; ++j) {
                mat->elementsPerDim[j] *= dims[i];
            }
        } else {
            for (int j = 0; j < numDimensions; ++j) {
                mat->elementsPerDim[j] = 1;
            }
        }
        mat->size *= dims[i];
    }

    mat->numDim = numDimensions;
    cudaMallocManaged(&(mat->shape), numDimensions * sizeof(int));
    cudaMemcpy((mat->shape), dims, numDimensions * sizeof(int), cudaMemcpyHostToDevice);

    return mat;
}

//Randomly initialized matrix
CudaMatrix* BuildRandomMatrix (int dims[], int numDimensions){
    CudaMatrix *mat = createGenMatrix(dims, numDimensions);

    // Initialize matrix
    cudaMallocManaged(&(mat->mat), mat->size * sizeof(int));
    cuRandArrInit<<<1, mat->size>>>(mat->mat);
    return mat;
}


CudaMatrix* BuildMatrixConst (const int dims[], int numDimensions, const int &c){
    CudaMatrix *mat = createGenMatrix(dims, numDimensions);

    // Initialize matrix
    cudaMalloc(&(mat->mat), mat->size * sizeof(int));
    cuConstArrInit<<<1, mat->size>>>(mat->mat, c);
    return mat;
}

CudaMatrix* BuildMatrixZeros (const int dims[], int numDimensions){
    return BuildMatrixConst(dims, numDimensions, 0);
}

void printMatrix(const CudaMatrix *m) {
    if (!m) return;
    //Currently only supports 2-D matrix operations
    int *local = (int *)malloc(m->size * sizeof(int));
    cudaMemcpy(local, m->mat, m->size * sizeof(int), cudaMemcpyDeviceToHost);
    printf("(%d, %d)\n", m->shape[0], m->shape[1]);
    int idx = 0;
    for (int i = 0; i < m->shape[0]; ++i){
        for (int j = 0; j < m->shape[1]; ++j){
            printf("%d\t", local[idx++]);
        }
        printf("\n");
    }
    free(local);
}


const int * getShape(CudaMatrix *mat) {
    return mat->shape;
}


__device__ int getIdx(CudaMatrix *m, const int *coords) {
    int arrIdx = 0;
    for (int i = 0; i < m->numDim; ++i){
    arrIdx += coords[i] * (*(m->elementsPerDim + i));
    }
    return arrIdx;
}

__device__ void setElement(CudaMatrix *m, const int*coords, const int &c){
    m->mat[getIdx(m, coords)] = c;
}

__device__ int &getElement(CudaMatrix *m, const int*coords){
    return m->mat[getIdx(m, coords)];
}
