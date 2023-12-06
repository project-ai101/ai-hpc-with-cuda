/*******************************************************************************
 *
 *     Copyright (c) 2023 Bin Tan
 *
 *******************************************************************************/

#ifndef _CUDA_MAT_MAT_MULTIPLY_HH_
#define _CUDA_MAT_MAT_MULTIPLY_HH_
#include <iostream>

#include "cuda_runtime.h"
#include "cuda.h"


#define BLOCK_SIZE 16

#define RETURN_ERROR -1
#define RETURN_SUCCESS 1

template<class ValType> class CudaMatMatMultiplyComp {
    public:
        enum MatrixIndex {
            A = 0,
            B,
            C,
            TOTAL_MATRIXES
        };
    protected:
        // Matrix Dimensions,
        //   m_ x k_ for A,
        //   k_ x n_ for B,
        //   m_ x n_ for C
        int m_, n_, k_;

        // matrix data pointers in host memory
        ValType* mat_host_[TOTAL_MATRIXES];
        // matrix data pointers in GPU device memory
        ValType* mat_dev_[TOTAL_MATRIXES];
        int matrix_size_[TOTAL_MATRIXES];
        int matrix_data_size_[TOTAL_MATRIXES];

        float alpha_;
        float beta_;


    public:
        CudaMatMatMultiplyComp(int m, int n, int k, float alpha, float beta);
        virtual ~CudaMatMatMultiplyComp();

        int init_mem();
        int load_data();
        int perform_comp();
        int retrieve_result();

        // getters to access matrix data memory
        ValType* getMatrixA() { return mat_host_[A]; }
        ValType* getMatrixB() { return mat_host_[B]; }
        ValType* getMatrixC() { return mat_host_[C]; }
};

#include "cuda_mat_mat_multiply.cpp"

#endif

