/*******************************************************************************
 *
 *     Copyright (c) 2023 Bin Tan
 *
 *******************************************************************************/

#include <iostream>
#include <cmath>

extern "C" {
    #include <sys/time.h>
}

#include "cuda_mat_mat_multiply.h"


int main(int argc, char** argv) {
   typedef CudaMatMatMultiplyComp<float> MultiplyCompFloat;

   int N = 4096;
   int M = 4096;
   int K = 4096;
   float alpha = 1.0f;
   float beta = 1.0f;
   MultiplyCompFloat* mulcomp = new MultiplyCompFloat(N, N, N, alpha, beta);


   if (mulcomp->init_mem() < 0) {
       delete mulcomp;
       return EXIT_FAILURE;
   }


   // initialize data
   float* a = mulcomp->getMatrixA();
   float* b = mulcomp->getMatrixB();
   float* c = mulcomp->getMatrixC();

   // a is a row-major matrix
   for (int row = 0; row < M; row++) {
       for (int col = 0; col < K; col++) {
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

   if (mulcomp->load_data() != RETURN_SUCCESS) {
       delete mulcomp;
       return EXIT_FAILURE;
   }

   struct timeval comp_start, comp_end;
   gettimeofday(&comp_start, nullptr);
   if (mulcomp->perform_comp() != RETURN_SUCCESS) {
       delete mulcomp;
       return EXIT_FAILURE;
   }

   if (mulcomp->retrieve_result() != RETURN_SUCCESS) {
       delete mulcomp;
       return EXIT_FAILURE;
   }
   gettimeofday(&comp_end, nullptr);

   float total_comp_time = (comp_end.tv_sec - comp_start.tv_sec) * 1000.0f +
                           (comp_end.tv_usec - comp_start.tv_usec) / 1000.0f;

   // validate result
   a = mulcomp->getMatrixA();
   b = mulcomp->getMatrixB();
   c = mulcomp->getMatrixC();
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

   delete mulcomp;

   std::cout << "Success" << std::endl;
   std::cout << "Total Comp Time of Matrix Matrix Multiplication with size ("
             << M << ", " << N << ", " << K << ") is "
             << total_comp_time << " milliseconds"
             << std::endl;

   return EXIT_SUCCESS;
}

