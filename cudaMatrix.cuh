#include <cuda_runtime.h>
#include <cstdio>
#include <random>

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
        for (int i = 0; i < size; ++i){
            array[i] = rand() % 20;
        }
    }

    void initDimensions(const int *dim_in) {
        int num_entries = 0;
        (int*) cudaMallocManaged(&dimensions, num_dim * sizeof(int));
        for (int i = 0; i < num_dim; ++i){
            dimensions[i] = dim_in[i];
            num_entries *= dimensions[i];
        }
        size = num_entries;
    }

    int elementsPerDim(int dim) {
        int res = 1;
        while (++dim < num_dim){
            res *= dimensions[dim];
        }
        return res;
    }

    int& getElement(const int* location){
        int start = 0;
        for (int i = 0; i < num_dim - 1; ++i){
            start += elementsPerDim(i) * location[i];
        }
        start += location[num_dim - 1];
        return array[start];
    }

    const int* getShape(int dim_number) {
        return dimensions;
    }

    // Currently Assuming we have a 2-D matrix for simplicity,
    // more dimensions will be added later
    void printMatrix() {
        int *hostArr = nullptr;
        int index = 0;
        cudaMemcpy(hostArr, array, size, cudaMemcpyDeviceToHost);
        for (int i = 0; i < dimensions[0]; ++i) {
            for (int j = 0; j < dimensions[1]; ++j) {
                // Access the element using the index and print it
                printf("%d ", hostArr[index]);
                index++;
            }
            printf("\n"); // Move to the next line after printing a row
        }
    }

    ~CudaMatrix() {
        cudaFree(dimensions);
        cudaFree(array);
    }
};

