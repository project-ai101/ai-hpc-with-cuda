/*******************************************************************************
 *
 *     Copyright (c) 2023 Bin Tan
 *
 *******************************************************************************/

#include <iostream>
#include <cmath>

extern "C" {
    #include <sys/time.h>
    #include <string.h>
}

#include "matrix_matrix_multiply.h"


int main(int argc, char** argv) {
   int M = 4096, N = 4096, K = 4096;
   float alpha = 1.0f, beta = 1.0f;

   MatrixMatrixMultiply* mmm = new MatrixMatrixMultiply(M, N, K);

   cutensorHandle_t handle;
   if (cutensorCreate(&handle) != CUTENSOR_STATUS_SUCCESS) {
       std::cout << "Failed to create CuTensor handle" << std::endl;
       return EXIT_FAILURE;
   }

   if (mmm->init(handle) < 0) {
       delete mmm;
       return EXIT_FAILURE;
   }


   // initialize data
   float* a = mmm->getA()->getHostMem();
   float* b = mmm->getB()->getHostMem();
   float* c = mmm->getC()->getHostMem();

   // a is a column-major matrix
   for (int col = 0; col < K; col++) {
       for (int row = 0; row < M; row++) {
           *a++ = (float) row;
       }
   }

   // b is a column-major matrixes
   for (int col = 0; col < N; col++) {
       for (int row = 0; row < K; row++) {
           *b++ = (float) (N - col);
       }
   }

   // c is a column-major matrixes
   for (int col = 0; col < N; col++) {
       for (int row = 0; row < M; row++) {
           *c++ = 0.0f;
       }
   }

   if (mmm->load_data() != RETURN_SUCCESS) {
       delete mmm;
       std::cout << "Failed to load data into device" << std::endl;
       return EXIT_FAILURE;
   }

   struct timeval comp_start, comp_end;
   gettimeofday(&comp_start, nullptr);
   if (mmm->perform_comp(alpha, beta) != RETURN_SUCCESS) {
       delete mmm;
       return EXIT_FAILURE;
   }
   if (mmm->retrieve_result() != RETURN_SUCCESS) {
       delete mmm;
       return EXIT_FAILURE;
   }
   gettimeofday(&comp_end, nullptr);

   float total_comp_time = (comp_end.tv_sec - comp_start.tv_sec) * 1000.0f +
                           (comp_end.tv_usec - comp_start.tv_usec) / 1000.0f;

   // validate result
   a = mmm->getA()->getHostMem();
   b = mmm->getB()->getHostMem();
   c = mmm->getC()->getHostMem();
   for (int col = 0; col < N; col++) {
       for (int row = 0; row < N; row++) {
           float expected = ((float)row) * ((float)(N - col)) * (float) K;
           float err_rate = std::abs(*c - expected);
           if (expected > 0) {
                err_rate /= expected;
           }
           if (err_rate > 0.0001) {
               std::cout << "Result is corrupted at ("
                         << row << ", " << col << ") expected "
                         << expected << " instead of "
                         << *c << " with error "
                         << err_rate
                         << std::endl;
               return EXIT_FAILURE;
           }
           c++;
       }
   }

   delete mmm;

   std::cout << "Success" << std::endl;
   std::cout << "cuTensor Matrix-Matrix-Multiplication - size ("
             << M << ", " << N << ", " << K << "), total comp time "
             << total_comp_time << " milliseconds"
             << std::endl;

   return EXIT_SUCCESS;
}
