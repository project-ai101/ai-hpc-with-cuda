/*******************************************************************************
 *
 *     Copyright (c) 2023 Bin Tan
 *
 *******************************************************************************/

#ifndef CUDA_BLAS_GEMV_COMP_HH_
#define CUDA_BLAS_GEMV_COMP_HH_

#include <cuda_runtime.h>
#include "cublas_v2.h"


//
// The cuBLAS gemv API performs the following matrix-vector multiplication
//            Y = alpha * op(A) * X + beta * Y
// where A is a M x N matrix, both X and Y are N dimensional vectors, and 
// both alpha and beta are scalar values.
//
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
#endif
