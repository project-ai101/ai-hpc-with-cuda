/*******************************************************************************
 *
 *     Copyright (c) 2023 Bin Tan
 *
 *******************************************************************************/
#include <iostream>
#include "cublas_gemv.h"

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
