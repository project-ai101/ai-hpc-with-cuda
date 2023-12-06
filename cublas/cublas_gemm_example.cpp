/*******************************************************************************
 *
 *     Copyright (c) 2023 Bin Tan
 *
 *******************************************************************************/

#include <string>
#include <iostream>
#include <cmath>

extern "C" {
    #include <sys/time.h>
}

#include "cublas_gemm.h"

void clean_comp(CuBlasGemmComp*& gemmComp, cublasHandle_t& handle, const char* err_msg) {
    cublasDestroy(handle);
    delete gemmComp;
    std::cout << err_msg << std::endl;
}

int main(int argc, char** argv) {
    int m = 128;
    int n = 128;
    int k = 128;
    cublasHandle_t handle;

    if (argc != 4) {
        std::cout << "Usage: ./cublas_gemm m n k" << std::endl;
        return 0;
    }

    try {
        m = std::stoi(argv[1]);
        n = std::stoi(argv[2]);
        k = std::stoi(argv[3]);
    } catch (std::invalid_argument const&  ia) {
        std::cout << "invalid argument: " << ia.what() << std::endl;
        return 0;
    } catch (std::out_of_range const& oor) {
        std::cout << "out of range: " << oor.what() << std::endl;
        return 0;
    }

    if (cublasCreate(&handle) != CUBLAS_STATUS_SUCCESS) {
        std::cout << "Failed to create a cublas handle\n";
        return 0;
    }

    if (cublasSetMathMode(handle, CUBLAS_PEDANTIC_MATH) != CUBLAS_STATUS_SUCCESS) {
        std::cout << "Failed to change cuBLAS Math Mode" << std::endl;
        cublasDestroy(handle);
        return 0;
    }

    int smCountTarget;
    cublasGetSmCountTarget(handle, &smCountTarget);
    std::cout << "SmCountTarget: " << smCountTarget << std::endl;

    if (cublasSetSmCountTarget(handle, 4) != CUBLAS_STATUS_SUCCESS) {
        std::cout << "Failed to to set SmCountTarget" << std::endl;
        cublasDestroy(handle);
        return 0;
    }

    cublasGetSmCountTarget(handle, &smCountTarget);
    std::cout << "SmCountTarget: " << smCountTarget << std::endl;

    CuBlasGemmComp* gemmComp = new CuBlasGemmComp(m, n, k, 1.0f, 1.0f);

    if (gemmComp->init_mem() < 0) {
        clean_comp(gemmComp, handle, "Failed to allocate memory for matrix and vectors");
        return 0;
    }

    // initialize the data in column-major layout
    float* a = gemmComp->getMatrix(CuBlasGemmComp::A);
    for (int col = 0; col < k; col++) {
        for (int row = 0; row < m; row++) {
            *a++ = (float) row;
        }
    }

    float* b = gemmComp->getMatrix(CuBlasGemmComp::B);
    for (int col= 0; col < n; col++) {
        for (int row = 0; row < k; row++) {
            *b++ = 1.0f;
        }
    }

    float* c = gemmComp->getMatrix(CuBlasGemmComp::C);
    for (int col = 0; col < n; col++) {
        for (int row = 0; row < m; row++) {
            *c++ = 0.0f;
        }
    }

    if (gemmComp->load_data() < 0) {
        clean_comp(gemmComp, handle, "Failed to load data from host to device");
        return 0;
    }

    struct timeval comp_start, comp_end;
    gettimeofday(&comp_start, nullptr);

    if (gemmComp->perform_comp(handle) < 0) {
        clean_comp(gemmComp, handle, "Failed to perofrm gemm computation");
        return 0;
    }

    if (gemmComp->fetch_result() < 0) {
        clean_comp(gemmComp, handle, "Failed to fetch result from device");
        return 0;
    }
    gettimeofday(&comp_end, nullptr);

    // validate the result
    c = gemmComp->getMatrix(CuBlasGemmComp::C);
    for (int col = 0; col < n; col++) {
        for (int row = 0; row < m; row++) {
            float expected_val = (float) row * k;
            float err = std::abs(*c - expected_val);
            if (err > 0.0001f) {
                std::cout << "The result " << *c
                          << " at index (" << row << ", " << col << ")"
                          << " does not match expected " << expected_val
                          << std::endl;
                clean_comp(gemmComp, handle, "CuBLAS gemm computation is incorrect");
                return 0;
            }
            c++;
        }
    }

    float total_time = (comp_end.tv_sec - comp_start.tv_sec) * 1000.0f +
                       (comp_end.tv_usec - comp_start.tv_usec) / 1000.0f;
    std::cout << "GEMM with size (" << m << ", " << n << ", " << k
              << ") took " << total_time << " ms" << std::endl;
    clean_comp(gemmComp, handle, "CuBLAS gemm computation is successful");
    return 0;
}
                                                                                                                                                                   239,1         Bot
