# cuBLAS Simple Example

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

The main cuBLAS API is used in this cublasSgemv. The API to perform the following computation task

```
Y = alpha * A * X + beta * Y
```
where A is a m x n matrix, X is an input vector with dimension n and Y is an input/output vector with dimension m.
Both alpha and beta are scalar float values.

### Headers Needed

```cpp
extern "C" {
    #include <cuda_runtime.h>
    #include "cublas_v2.h>
}
```
