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

void clean_comp(CuBlasGemvComp*& gemvComp, cublasHandle_t& handle, const char* err_msg) {
    cublasDestroy(handle);
    delete gemvComp;
    std::cout << err_msg << std::endl;
}

int main(int argc, char** argv) {
    cublasHandle_t handle;

    if (cublasCreate(&handle) != CUBLAS_STATUS_SUCCESS) {
        std::cout << "Failed to create a cublas handle\n";
        return 0;
    }

    int rows = 128;
    int cols = 128;
    CuBlasGemvComp* gemvComp = new CuBlasGemvComp(rows, cols, 1.0f, 1.0f);

    if (gemvComp->init_mem() < 0) {
        clean_comp(gemvComp, handle, "Failed to allocate memory for matrix and vectors");
        return 0;
    }

    // initialize the data in column-major layout
    float* a = gemvComp->getA();
    for (int col = 0; col < cols; col++) {
        for (int row = 0; row < rows; row++) {
            a[col * rows + row] = (float) row;
        }
    }

    float* x = gemvComp->getX();
    for (int i = 0; i < cols; i++) {
        x[i] = 1.0f;
    }

    float* y = gemvComp->getY();
    for (int i = 0; i < rows; i++) {
        y[i] = 0.0f;
    }

    if (gemvComp->load_data() < 0) {
        clean_comp(gemvComp, handle, "Failed to load data from host to device");
        return 0;
    }

    if (gemvComp->perform_comp(handle) < 0) {
        clean_comp(gemvComp, handle, "Failed to perofrm gemv computation");
        return 0;
    }

    if (gemvComp->fetch_result() < 0) {
        clean_comp(gemvComp, handle, "Failed to fetch result from device");
        return 0;
    }

    // validate the result
    for (int i = 0; i < rows; i++) {
        float expected_val = (float) i * cols;
        if (y[i] != expected_val) {
            std::cout << "The result " << y[i] << " at index " << i
                      << " does not match expected " << expected_val << std::endl;
            clean_comp(gemvComp, handle, "CuBLAS gemv computation is incorrect");
            return 0;
        }
    }

    clean_comp(gemvComp, handle, "CuBLAS gemv computation is successful");
    return 0;
}
    
