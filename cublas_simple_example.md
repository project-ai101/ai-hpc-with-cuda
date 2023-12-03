# cuBLAS Simple Example in C++

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
### cuBLAS API

The main cuBLAS API is used is [cublasSgemv](https://docs.nvidia.com/cuda/cublas/index.html#cublas-t-gemv). 

```
cublasStatus_t cublasSgem(cublasHandle_t h, cublasOperation_t transposeOps, int m, int n,
                          const float* alpha, const float* A, int lda,
                          const float* x, int incrementX, const float* beta,
                          const float* y, int incrementY);
```
This API performs the following computation task

```
y = alpha * A * x + beta * y
```
where A is a m x n matrix, x is an input vector with dimension n and y is an input/output vector with dimension m.
Both alpha and beta are scalar float values.

### Headers Needed

```cpp
extern "C" {
    #include <cuda_runtime.h>
    #include "cublas_v2.h>
}
```

### Class Design

The normal CUDA computation flow is
  1. Allocate host and device memory for computation data
  2. Initialize the data and load them from host to device
  3. Perform the computation in the GPU devices
  4. Fetch the computation results from the GPU devices

Design the following class to capture the common behaviors,

```cpp
class CuBlasGemvComp {
    protected:
        int M_, N_;         // matrix dimensions where M_ for rows and N_ for cols
        float* a_host_;     // host memory addresses of the Matrix A
        float* a_dev_;      // GPU device memory address of the Matrix A

        float* x_host_;     // host memory address of the input vector X
        float* x_dev_;      // GPU device memory address of the input vector X

        float* y_host_;     // host memory address of the input/output vector Y
        float* y_dev_;      // GPU device memory address of the input/output vector Y

        float alpha_;
        float beta_;

    public:
        CuBlasGemvComp(int rows, int cols, float alpha, float beta);
        virtual ~CuBlasGemvComp();

        // allocate memory spaces for metrix A and vector X and Y in both host and device
        int init_mem();

        // load initialized host A, X and Y into device memory
        int load_data();

        // invoke the cublas gemv API
        int perform_comp(cublasHandle_t& handle);

        // fetch the output result in Y device memory into Y host memory
        int fetch_result();

        // getters
        float* getA() { return a_host_; }
        float* getX() { return x_host_; }
        float* getY() { return y_host_; }
};

```

### Implementation Analysis

The constructor,
```cpp
        CuBlasGemvComp(int rows, int cols, float alpha, float beta);
```
mainly initialize all pointers into nullptr but not allocate memory. 
The reason is to simplify the error handling and memory allocation may not be always successful.
Therefore, the memory allocation is separated into its own method,

```cpp
        int init_mem();
```
malloc is used for host memory allocation and cudaMalloc is used for device side memory allocation.

After the memory are allocated, we need to access them to initialize with proper data and read out results.
The getters are designed for this purpose

```cpp
        float* getA() { return a_host_; }
        float* getX() { return x_host_; }
        float* getY() { return y_host_; }
```

### C++ Code

[cublas_gemv.h](./cublas_gemv.h), [cublas_gemv.cpp](./cublas_gemv.cpp) and [cublas_gemv_example.cpp](./cublas_gemv_example.cpp)

### Compile and Run

```
$ nvcc -I${CUBLAS_INC_PATH} -lcublas cublas_gemv.cpp cublas_gemv_example.cpp -o cublas_gemv_example
$ ./cublas_gemv_example
CuBLAS gemv computation is successful
```
