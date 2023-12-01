cmake_minimum_required(VERSION 3.26)
project(cuda_LP CUDA)

set(CMAKE_CUDA_STANDARD 17)

# Set the architecture according to your GPU (e.g., 6.1 for Pascal architecture)
set(CMAKE_CUDA_ARCHITECTURES 75)

# Add additional flags if needed
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler -Wall")

add_executable(cuda_LP main.cu
        cudaMatrix.cuh
        matFxns.cuh
        cudaMatrix.cu
)

set_target_properties(cuda_LP PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
)
