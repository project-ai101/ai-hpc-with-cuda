# cuBLAS Matrix-Mtrix Multiplication Example in C++

### Environment Setup
The platform which is used in this example is Ubuntu 22.04.

Follow [this document](./cuda_env_setup.md) to check if the CUDA environment is setup.

If the cuBLAS library is not installed, follow this [link](https://developer.nvidia.com/cublas) to download and install it.

Then export following environment variables after the cuBLAS library is installed,

```
export CUBLAS_ROOT=<cublas lib parent directory>
export CUBLAS_LIB_PATH=${CUBLAS_ROOT}/lib64
export CUBLAS_INC_PATH=${CUBLAS_ROOT}/include
export LD_LIBRARY_PATH=${CUBLAS_LIB_PATH}:${LD_LIBRARY_PATH}
```
### cuBLAS Matrix-Matrix Multiplication API

The main cuBLAS Matrix-Matrix Multiplication API is used is [cublasSgemm](https://docs.nvidia.com/cuda/cublas/index.html#cublas-t-gemm). 

```
cublasStatus_t cublasSgemm(cublasHandle_t h,
                           cublasOperation_t transposeOpsA, cublasOperation_t transposeOpsB,
                           int m, int n, int k,
                           const float* alpha, const float* A, int lda, const float* B, int ldb,
                           const float* beta, const float* C, int ldc);
```
This API performs the following computation task

```
C = alpha * A * B + beta * C
```
where A is a m x k matrix, B is a k x n matrix and C is a m x n matrix. Both alpha and beta are scalar float values.

### Headers Needed

```cpp
extern "C" {
    #include <cuda_runtime.h>
    #include "cublas_v2.h>
}
```

### Class Design

Following the same design principle in the [cuBLAS Matrix Vector example](./cublas_matrix_vector_multiplication_example.md), design the CuBlasGemmComp class as,

```cpp
class CuBlasGemmComp {
    public:
        enum MatrixIndex {
            A = 0,
            B,
            C,
            TOTAL_MATRICES
        };
    protected:
        int M_, N_, K_;                       // matrix dimension

        float* host_mem_base_;                // the base host memory addresses for MATRICES
        float* mat_host_[TOTAL_MATRICES];     // host memory addresses of the MATRICES
        float* dev_mem_base_;                 // the base GPU device memory address  for MATRICES
        float* mat_dev_[TOTAL_MATRICES];      // GPU device memory address of the MATRICES

        float alpha_;
        float beta_;

    public:
        CuBlasGemmComp(int& m, int& n, int& k, float alpha, float beta);
        virtual ~CuBlasGemmComp();

        // allocate memory spaces for metrix A and vector X and Y in both host and device
        int init_mem();

        // load initialized host A, X and Y into device memory
        int load_data();

        // invoke cublas gemv computation
        int perform_comp(cublasHandle_t& handle);

        // fetch the output result in Y device memory into Y host memory
        int fetch_result();

        // getter
        float* getMatrix(MatrixIndex idx) { return mat_host_[idx]; }
};
#endif
```

### Implementation Analysis

The constructor,
```cpp
        CuBlasGemmComp(int m, int n, int k, float alpha, float beta);
```
mainly initialize all pointers into nullptr but not allocate memory. 
The reason is to simplify the error handling and memory allocation may not be always successful.
Therefore, the memory allocation is separated into its own method,

```cpp
        int init_mem();
```
malloc is used for host memory allocation and cudaMalloc is used for device side memory allocation.

After the memory are allocated, we need to access them to initialize with proper data and read out results.
The getter are designed for this purpose

```cpp
        float* getMatrix(MatrixIndex idx) { return mat_host_[idx]; }
```

### C++ Code

[cublas_gemm.h](./cublas_gemm.h), [cublas_gemm.cpp](./cublas_gemm.cpp) and [cublas_gemm_example.cpp](./cublas_gemm_example.cpp)

### Compile and Run

```
$ nvcc -I${CUBLAS_INC_PATH} -lcublas cublas_gemm.cpp cublas_gemm_example.cpp -o cublas_gemm_example
$ ./cublas_gemm_example 4096 4096 4096
GEMM with size (4096, 4096, 4096) took 28.772 ms
CuBLAS gemm computation is successful
```
