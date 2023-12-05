# CUDA Matrix-Matrix Addition in C++
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; -- Author: Bin Tan

The C implementaiton of matrix-matrix addition is in [cuda_matrix_add.cu](./cuda_matrix_add.cu). In
this document, the C++ implementation of the matrix-matrix addtion is discussed.

### Class Design
  
A normal CUDA computation flow is
  1. Allocate host and device memory for computation data
  2. Initialize data
  3. Load data from host to device
  4. Perform the computation in the GPU devices
  5. Fetch the computation results from the GPU devices
  6. Valid the computation results

As there are infinite many different data, initializing data is real content based and not a 
common behavior. Therefore, it shall be excluded from the class design.

With validatino of the computation results, one common implementation is to re-compute the results
with CPUs and then compare them with respect to the result from GPUs. However, for large dimension
matrixes, the computation over CPUs is very slow. Therefore, the validation is also excluded from
the class design so that some specific data content can be designed by validation.

A CUDA computation is also bounded to a particular data type, such as, float, double, int, etc.,
though the computatoin algorithm is identical. To avoid to re-implement the same code repeatly,
C++ template is adopted.

Therefore, the following class is designed to capture the common behaviors and implementation
for various data types,

```cpp
template<class ValType> class CudaMatMatAddComp {
    public:
        enum MatrixIndex {
            A = 0,
            B,
            C,
            TOTAL_MATRIXES
        };
    protected:
        // Matrix Dimension m_ x n_
        int m_, n_;

        // Matrix Size: total elements
        int matrix_size_;

        // Matrix Data Size: total elements * sizeof(ValType)
        int matrix_data_size_;

        // matrix data pointers in host memory
        ValType* mat_host_[TOTAL_MATRIXES];

        // matrix data pointers in GPU device memory
        ValType* mat_dev_[TOTAL_MATRIXES];

    public:
        CudaMatMatAddComp(int m, int n);
        virtual ~CudaMatMatAddComp();

        int init_mem();
        int load_data();
        int perform_comp();
        int retrieve_result();

        // getters to access matrix data memory
        ValType* getMatrixA() { return mat_host_[A]; }
        ValType* getMatrixB() { return mat_host_[B]; }
        ValType* getMatrixC() { return mat_host_[C]; }
};
```

### C++ Implementation

- [cuda_mat_mat_add.h](./cuda_mat_mat_add.h)
- [cuda_mat_mat_add.cpp](./cuda_mat_mat_add.cpp)
- [cuda_mat_mat_add_example.cu](./cuda_mat_mat_add_example.cu)
