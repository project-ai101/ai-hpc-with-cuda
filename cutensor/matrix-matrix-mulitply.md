# Matrix-Matrix Multiplication in C++ with cuTENSOR
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; -- Author: Bin Tan

### Overview
cuTENSOR is a Nvidia GPU accelerated library from Nvidia for tensor contraction, reduction and element-wise operations.
Consider a matrix as a tensor with order 2. Two such tensors form a tensor with order 4. Denote them as
$` A_{i, k} \otimes B^{k, j} `$. The contraction then becomes as the matrix-matrix multiplication by summation the 
modes (aka indexes) $` i `$. 

The version of the cuTENSOR library which was used in this example is 2.0.0.7.

### Class Design
Programming over the cuTENSOR library is not trivial. Lots of preparation are needed before a computation operation 
is performed. To make the implementation is reusable and easy to use, object-oriented approach is adopted. To hide 
the complexity of the detail implementation behind an API, encapsulation from Class is a good solution. Further,
object-oriented approach also provides an efficient to modularize a complicated task, such as tensor contraction.
With these two principles, the tensor contraction task is designed into two classes, MatrixTensor and MatrixMatrixMultiplication.
MatrixTensor mainly encapsulates the data manipulation while MatrixMatrixMultiplication focuses on invoking the
cuTENSOR contraction API.

```cpp
class MatrixTensor {
    protected:
        enum ModeIndex {
            ROW = 0,
            COL,
            TOTAL_MODES
        };
        cutensorHandle_t handle_;
        cutensorDataType_t cu_data_type_;
        cutensorTensorDescriptor_t descriptor_;

        int modes_[TOTAL_MODES];
        long extends_[TOTAL_MODES];
        unsigned int data_len_;                    // in bytes
        float* host_mem_;
        float* dev_mem_;

        // allow MatrixMatrixMultiply function object to access these APIs
        friend class MatrixMatrixMultiply;
        int load_to_device();
        int retrieve_from_device();
        float* getDeviceMem() { return dev_mem_; }

        MatrixTensor(int row_mode, int row_extend, int col_mode, int col_extend);
        int init(cutensorHandle_t& handle);
    public:
        virtual ~MatrixTensor();
        // column-major
        unsigned int getRows() { return extends_[ROW]; }
        unsigned int getCols() { return extends_[COL]; }
        const int* getModes() { return modes_; }
        const long* getExtends() { return extends_; }

        cutensorDataType_t getDataType() { return cu_data_type_; }
        cutensorTensorDescriptor_t getDescriptor() { return descriptor_; }
        float* getHostMem() { return host_mem_; }
};

class MatrixMatrixMultiply {
    protected:
        enum TensorIndex {
            A = 0,
            B,
            C,
            TOTAL_TENSORS
        };

        cutensorHandle_t handle_;
        MatrixTensor* tensors_[TOTAL_TENSORS];
        cutensorOperationDescriptor_t ops_desc_;
        cutensorComputeDescriptor_t comp_desc_;
        cutensorAlgo_t algorithm_;
        cutensorWorksizePreference_t worksize_pref_;
        void* actual_workspace_;
        unsigned long actual_workspace_size_;
        unsigned long estimate_workspace_size_;

        cutensorPlanPreference_t plan_pref_;
        cutensorPlan_t plan_;
    public:
        MatrixMatrixMultiply(int m, int n, int k);
        virtual ~MatrixMatrixMultiply();

        int init(cutensorHandle_t& handle);
        int load_data();
        int perform_comp(float alpha, float beta);
        int retrieve_result();

        MatrixTensor* getA() { return tensors_[A]; }
        MatrixTensor* getB() { return tensors_[B]; }
        MatrixTensor* getC() { return tensors_[C]; }
};
```

### C++ Files

- [matrix_matrix_multiply.h](./matrix_matrix_multiply.h)
- [matrix_matrix_multiply.cpp](./matrix_matrix_multiply.cpp)
- [matrix_matrix_multiply_example.cu](./matrix_matrix_multiply_example.cu)

### Compile and Run
Use the following command to compile the example code,

```
$ nvcc -std=c++11 -I${CUTENSOR_INC_PATH} -L${CUTENSOR_LIB_PATH} matrix_matrix_multiply.cpp matrix_matrix_multiply_example.cu -lcutensor -lcudart -o matrix_matrix_multiply
```

CUTENSOR_INC_PATH is the path to the file directory which has all cuTENSOR include header files and CUTENSOR_LIB_PATH 
is the path to the file directory which has all cuTENSOR libraries.

Use the following command to run the example code,
```
$ ./matrix_matrix_multiply
```

The output is 
```
Success
cuTensor Matrix-Matrix-Multiplication - size (4096, 4096, 4096), total comp time 23.413 milliseconds
```
