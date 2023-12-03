/*******************************************************************************
 *
 *     Copyright (c) 2023 Bin Tan
 *
 *******************************************************************************/

#include <string>
#include <iostream>
#include "cublas_gemm.h"

CuBlasGemmComp::CuBlasGemmComp(int& m, int& n, int& k, float alpha, float beta) {
    M_ = m;
    N_ = n;
    K_ = k;
    alpha_ = alpha;
    beta_ = beta;

    host_mem_base_ = nullptr;
    dev_mem_base_  = nullptr;
    for (int i = 0; i < TOTAL_MATRIXES; i++) {
        mat_host_[i] = nullptr;
        mat_dev_[i]  = nullptr;
    }
}

CuBlasGemmComp::~CuBlasGemmComp() {
    if (host_mem_base_ != nullptr) { delete[] host_mem_base_; }
    if (dev_mem_base_ != nullptr) { cudaFree(dev_mem_base_); }
}

int CuBlasGemmComp::init_mem() {
    int mat_len[TOTAL_MATRIXES] {
        M_ * K_,
        K_ * N_,
        M_ * N_
    };

    int total_mem_len = 0;
    for (int i = 0; i < TOTAL_MATRIXES; i++) {
        total_mem_len += mat_len[i];
    }
    host_mem_base_ = new float[total_mem_len];
    if (cudaMalloc((void**)&dev_mem_base_, total_mem_len * sizeof(float)) != cudaSuccess) {
        return -1;
    }

    float* host_mem_addr = host_mem_base_;
    float* dev_mem_addr = dev_mem_base_;
    for (int i = 0; i < TOTAL_MATRIXES; i++) {
        mat_host_[i] = host_mem_addr;
        mat_dev_[i] = dev_mem_addr;
        host_mem_addr += mat_len[i];
        dev_mem_addr += mat_len[i];
    }

    return 1;
}


int CuBlasGemmComp::load_data() {
    if (cublasSetMatrix(M_, K_, sizeof(float), mat_host_[0], M_, mat_dev_[0], M_) != CUBLAS_STATUS_SUCCESS) {
        std::cout << "failed to load Matrix A" << std::endl;
        return -1;
    }

    if (cublasSetMatrix(K_, N_, sizeof(float), mat_host_[1], K_, mat_dev_[1], K_) != CUBLAS_STATUS_SUCCESS) {
        std::cout << "failed to load Matrix B" << std::endl;
        return -1;
    }

    if (cublasSetMatrix(M_, N_, sizeof(float), mat_host_[2], M_, mat_dev_[2], M_) != CUBLAS_STATUS_SUCCESS) {
        std::cout << "failed to load Matrix C" << std::endl;
        return -1;
    }

    return 1;
}


int CuBlasGemmComp::perform_comp(cublasHandle_t& handle) {
    cublasStatus_t stat = cublasSgemm(handle,
                                      CUBLAS_OP_N,
                                      CUBLAS_OP_N,
                                      M_,
                                      N_,
                                      K_,
                                      &alpha_,
                                      mat_dev_[0],
                                      M_,
                                      mat_dev_[1],
                                      K_,
                                      &beta_,
                                      mat_dev_[2],
                                      M_);

    if (stat != CUBLAS_STATUS_SUCCESS) {
        return -1;
    }

    return 1;
}

int CuBlasGemmComp::fetch_result() {
    cublasStatus_t stat = cublasGetMatrix(M_, N_, sizeof(float), mat_dev_[2], M_, mat_host_[2], M_);
    if (stat != CUBLAS_STATUS_SUCCESS) {
        return -1;
    }

    return 1;
}
