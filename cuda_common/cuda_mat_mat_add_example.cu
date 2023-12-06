/*******************************************************************************
 *
 *     Copyright (c) 2023 Bin Tan
 *
 *******************************************************************************/

#include <iostream>

#include "cuda_mat_mat_add.h"


int main(int argc, char** argv) {
   typedef CudaMatMatAddComp<float> AddCompFloat;

   int N = 4096;
   AddCompFloat* addcomp = new AddCompFloat(N, N);


   if (addcomp->init_mem() < 0) {
       delete addcomp;
       return EXIT_FAILURE;
   }


   // initialize data
   float* a = addcomp->getMatrixA();
   float* b = addcomp->getMatrixB();
   float* c = addcomp->getMatrixC();
   for (int col = 0; col < N; col++) {
       for (int row = 0; row < N; row++) {
           *a++ = (float) (col * N + row);
           *b++ = (float) (row * N + col);
           *c++ = 0.0f;
       }
   }

   if (addcomp->load_data() != RETURN_SUCCESS) {
       delete addcomp;
       return EXIT_FAILURE;
   }

   if (addcomp->perform_comp() != RETURN_SUCCESS) {
       delete addcomp;
       return EXIT_FAILURE;
   }

   if (addcomp->retrieve_result() != RETURN_SUCCESS) {
       delete addcomp;
       return EXIT_FAILURE;
   }

   // validate result
   a = addcomp->getMatrixA();
   b = addcomp->getMatrixB();
   c = addcomp->getMatrixC();
   for (int col = 0; col < N; col++) {
       for (int row = 0; row < N; row++) {
           float expected = (*a++) + (*b++);
           if (*c != expected) {
               std::cout << "Result is corrupted at ("
                         << row << ", " << col << ") expected "
                         << expected << " instead of "
                         << *c << std::endl;
               return EXIT_FAILURE;
           }
           c++;
       }
   }

   delete addcomp;

   std::cout << "Success" << std::endl;
   return EXIT_SUCCESS;
}
