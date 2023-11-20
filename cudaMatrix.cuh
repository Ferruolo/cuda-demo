#include <cuda_runtime.h>
#include <cstdio>
#include <random>
#include <curand_kernel.h>

__global__ void cuRandArr(int *randArray) {
    int tid = threadIdx.x;
    curandState state;
    curand_init(clock64(), tid, 0, &state);
    int r = (int)(curand_uniform(&state) * 20);
    randArray[tid] = (r);
}

class CudaMatrix {
private:
    int num_dim = 0;
    int size = 0;
    int *dimensions = nullptr;
    int *array = nullptr;
public:
    CudaMatrix(int num_dim_in, const int *dim_in, const int *arr_in) : num_dim(num_dim_in){
        // MallocNewMatrix
        initDimensions(dim_in);
        cudaMalloc(&array, size * sizeof(int));
        for (int i = 0; i < size; ++i){
            array[i] = arr_in[i];
        }
    }



    CudaMatrix(int num_dim_in, const int *dim_in) : num_dim(num_dim_in) {
        initDimensions(dim_in);

        cudaMalloc(&array, size * sizeof(int));
        cuRandArr<<<1, size>>>(array);
    }

    void initDimensions(const int *dim_in) {
        size = 1;
        (int*) cudaMallocManaged(&dimensions, num_dim * sizeof(int));
        for (int i = 0; i < num_dim; ++i){
            dimensions[i] = dim_in[i];
            size *= dim_in[i];
        }
    }

    __device__ int elementsPerDim(int dim) {
        int res = 1;
        while (++dim < num_dim){
            res *= dimensions[dim];
        }
        return res;
    }

    __device__ int& getElement(const int* location){
        int start = 0;
        for (int i = 0; i < num_dim - 1; ++i){
            start += elementsPerDim(i) * location[i];
        }
        start += location[num_dim - 1];
        return array[start];
    }

    const int* getShape() {
        return dimensions;
    }

    // Currently Assuming we have a 2-D matrix for simplicity,
    // more dimensions will be added later
     void printMatrix() {
        int *hostArr = (int *)malloc(size * sizeof(int));
        cudaMemcpy(hostArr, array, size * sizeof(int), cudaMemcpyDeviceToHost);

        printf("(%d, %d)\n", dimensions[0], dimensions[1]);
        for (int i = 0; i < dimensions[0]; ++i) {
            for (int j = 0; j < dimensions[1]; ++j) {
                // Access the element using the correct index calculation
                int index = i * dimensions[1] + j;
                printf("%d ", hostArr[index]);
            }
            printf("\n");
        }

        free(hostArr);
    }

    ~CudaMatrix() {
        cudaFree(dimensions);
        cudaFree(array);
    }
};

