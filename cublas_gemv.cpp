/*******************************************************************************
 *
 *     Copyright (c) 2023 Bin Tan
 *
 *******************************************************************************/
#include <iostream>
#include "cublas_gemv.h"


CuBlasGemvComp::CuBlasGemvComp(int rows, int cols, float alpha, float beta) {
    M_ = rows;
    N_ = cols;
    alpha_ = alpha;
    beta_ = beta;

    a_host_ = nullptr;
    a_dev_  = nullptr;
    x_host_ = nullptr;
    x_dev_  = nullptr;
    y_host_ = nullptr;
    y_dev_  = nullptr;
}

CuBlasGemvComp::~CuBlasGemvComp() {
    if (a_host_ != nullptr) { delete[] a_host_; }
    if (x_host_ != nullptr) { delete[] x_host_; }
    if (y_host_ != nullptr) { delete[] y_host_; }

    if (a_dev_ != nullptr) { cudaFree(a_host_); }
    if (x_dev_ != nullptr) { cudaFree(x_dev_); }
    if (y_dev_ != nullptr) { cudaFree(y_dev_); }
}

int CuBlasGemvComp::init_mem() {
    a_host_ = new float[M_ * N_ * sizeof(float)];
    x_host_ = new float[N_ * sizeof(float)];
    y_host_ = new float[M_ * sizeof(float)];

    if (cudaMalloc((void**)&a_dev_, M_*N_*sizeof(*a_dev_)) != cudaSuccess) {
        return -1;
    }

    if (cudaMalloc ((void**)&x_dev_, N_*sizeof(*x_dev_)) != cudaSuccess) {
        return -1;
    }

    if (cudaMalloc ((void**)&y_dev_, M_*sizeof(*y_dev_)) != cudaSuccess) {
        return -1;
    }

    return 1;
}


int CuBlasGemvComp::load_data() {
    int i;

    if (cublasSetMatrix(M_, N_, sizeof(*a_host_), a_host_, M_, a_dev_, M_) != CUBLAS_STATUS_SUCCESS) {
        return -1;
    }

    if (cublasSetVector(N_, sizeof(*x_host_), x_host_, 1, x_dev_, 1) != CUBLAS_STATUS_SUCCESS) {
        return -1;
    }

    if (cublasSetVector(M_, sizeof(*y_host_), y_host_, 1, y_dev_, 1) != CUBLAS_STATUS_SUCCESS) {
        return -1;
    }

    return 1;
}

int CuBlasGemvComp::perform_comp(cublasHandle_t& handle) {
    cublasStatus_t stat = cublasSgemv(handle,
                                      CUBLAS_OP_N,
                                      M_,
                                      N_,
                                      &alpha_,
                                      a_dev_,
                                      N_,
                                      x_dev_,
                                      1,
                                      &beta_,
                                      y_dev_,
                                      1);

    if (stat != CUBLAS_STATUS_SUCCESS) {
        return -1;
    }

    return 1;
}


int CuBlasGemvComp::fetch_result() {
    cublasStatus_t stat = cublasGetVector(M_, sizeof(*y_dev_), y_dev_, 1, y_host_, 1);
    if (stat != CUBLAS_STATUS_SUCCESS) {
        return -1;
    }

    return 1;
}
