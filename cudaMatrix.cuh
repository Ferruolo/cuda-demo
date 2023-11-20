#include <cuda_runtime.h>
#include <cstdio>


class CudaMatrix {
private:
    int num_dim = 0;
    int *dimensions = nullptr;
    int *array = nullptr;
public:
    CudaMatrix(int num_dim_in, int *dim_in, int *arr_in) : num_dim(num_dim_in){
        // MallocNewMatrix
        int num_entries = 0;
        (int*) cudaMallocManaged(&dimensions, num_dim * sizeof(int));
        for (int i = 0; i < num_dim; ++i){
            dimensions[i] = dim_in[i];
            num_entries *= dimensions[i];
        }
        cudaMalloc(&array, num_entries * sizeof(int));
        for (int i = 0; i < num_entries; ++i){
            array[i] = arr_in[i];
        }
    }




    ~CudaMatrix() {
        cudaFree(dimensions);
    }

    int getDimensions(int dim_number) {

    }


};

