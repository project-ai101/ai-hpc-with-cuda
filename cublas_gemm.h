/*******************************************************************************
 *
 *     Copyright (c) 2023 Bin Tan
 *
 *******************************************************************************/

#ifndef CUDA_BLAS_GEMM_COMP_HH_
#define CUDA_BLAS_GEMM_COMP_HH_

#include <cuda_runtime.h>
#include "cublas_v2.h"


//
// The cuBLAS gemm API performs the following matrix-matrix multiplication
//            C = alpha * op(A) * op(B) + beta * C
// where A is a M x K matrix, B is a K x N matrix, C is a M x N matrix and 
// both alpha and beta are scalar values.
//
class CuBlasGemmComp {
    public:
        enum MatrixIndex {
            A = 0,
            B,
            C,
            TOTAL_MATRIXES
        };
    protected:
        int M_, N_, K_;                       // matrix dimension

        float* host_mem_base_;                // the base host memory addresses for Matrixes
        float* mat_host_[TOTAL_MATRIXES];     // host memory addresses of the Matrixes
        float* dev_mem_base_;                 // the base GPU device memory address  for Matrixes
        float* mat_dev_[TOTAL_MATRIXES];      // GPU device memory address of the Matrixes

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

        // getters
        float* getMatrix(MatrixIndex idx) { return mat_host_[idx]; }
};
#endif
