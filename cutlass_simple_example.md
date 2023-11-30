# cuBLAS Simple Example

### Environment Setup
To use cuBLAS APIs, a Nvidia GPU device with tensor cores shall be available. A proper CUDA toolkit, including a nvcc compiler, shall be installed. 
Then follow this [link](https://developer.nvidia.com/cublas) to download the cuBLAS library.

On a Linux box, export following environment variables after the cuBLAS library is installed,

```
export CUBLAS_ROOT=<cublas lib parent directory>
export CUBLAS_LIB_PATH=${CUBLAS_ROOT}/lib64
export CUBLAS_INC_PATH=${CUBLAS_ROOT}/include
export LD_LIBRARY_PATH=${CUBLAS_LIB_PATH}:${LD_LIBRARY_PATH}
```
### Computation Task
```
C = A * B + C
```
where A is a m x k matrix, B is a k x n matrix and C is a m x n matrix.

The elements of A and B can half float or float type. The elements of C have type float.

### A Brief Design Analysis
Reusability is in the heart of software development. To support both half float and float computation,
C++ template technique is a good solution. 

Since most of CPUs do not support half float computation, helper functions over GPUs t
o convert a float matrix into a half float matrix are needed.

Therefore, the implementation can be simply designed into a 4-stage functional C++ implementation,
  - data preparation
  - cuBLAS APIs invocation
  - result data retrieval
  - validation

### Headers Needed

```cpp
#include <iostream>

extern "C" {
    #include <cuda_runtime.h>
    #include "cublas_v2.h>
}
```

### Implementation
#### Data Structure
To make code modulized and reusable, create a data struct CompData to hold all required data for a cuBLAS computation.

```cpp
    struct CompData {
        float* A_host_;        // the host memory address of the matrix A
        float* B_host_;        // the host memory address of the matrix B
        float* C_host_;        // the host memory address of the matrix C

        float* A_device_;      // the GPU device memory address of the matrix A
        float* B_device_;      // the GPU device memory address of the matrix B
        float* C_device_;      // the GPU device memory address of the matrix C

        half*  A_device_half_; // the GPU device memory address of the matrix A with half float type
        half*  B_device_half_; // the GPU device memory address of the matrix B with half float type

        // Matrix dimensions. A is a M_ x K_ matrix, B is a K_ x N_ matrix and C is a M_ x N_ matrix
        int M_;
        int N_;
        int K_;
    };
```
