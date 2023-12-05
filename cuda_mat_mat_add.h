/*******************************************************************************
 *
 *     Copyright (c) 2023 Bin Tan
 *
 *******************************************************************************/

#ifndef _CUDA_MAT_MAT_ADD_HH_
#define _CUDA_MAT_MAT_ADD_HH_
#include <iostream>

#include "cuda_runtime.h"
#include "cuda.h"

#define BLOCK_DIM_X 16
#define BLOCK_DIM_Y 16

#define RETURN_ERROR -1
#define RETURN_SUCCESS 1

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

#include "cuda_mat_mat_add.cpp"

#endif

