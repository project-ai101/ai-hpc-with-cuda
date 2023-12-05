/*******************************************************************************
 *
 *     Copyright (c) 2023 Bin Tan
 *
 *******************************************************************************/

#include <stdio.h>
#include <cuda_runtime.h>
#include "cuda.h"


__global__ void cuda_mat_add(float* a, float* b, float *c) {
   int element_x = blockIdx.x * blockDim.x + threadIdx.x;
   int element_y = blockIdx.y * blockDim.y + threadIdx.y;
   int column_len = gridDim.y * blockDim.y;

   int element_id = element_x * column_len + element_y;
   c[element_id] = a[element_id] + b[element_id];
}


int main(int argc, char** argv) {
   int total_matrixes = 3;
   int N = 4096;
   int matrix_data_size = N * N * sizeof(float);

   float* mat_dev [total_matrixes] {nullptr, nullptr, nullptr};
   float* mat_host[total_matrixes] {nullptr, nullptr, nullptr};

   // allocate device side matrix memory
   for (int i = 0; i < total_matrixes; i++) {
        if (cudaMalloc((void**)&mat_dev[i], matrix_data_size) != cudaSuccess) {
            for (int j = 0; j < total_matrixes; j++) {
                 if (mat_dev[j] != nullptr) {
                     cudaFree(mat_dev[j]);
                 }
            }
            printf("Failed to allocated device memory for matrix %d\n", i);
            return EXIT_FAILURE;
        }
   }

   // allocate host side matrix memory
   for (int i = 0; i < total_matrixes; i++) {
        mat_host[i] = (float*)malloc(matrix_data_size);
   }

   // initialize data
   float* a = mat_host[0];
   float* b = mat_host[1];
   float* c = mat_host[2];
   for (int col = 0; col < N; col++) {
       for (int row = 0; row < N; row++) {
           *a++ = (float) (col * N + row);
           *b++ = (float) (row * N + col);
           *c++ = 0.0f;
       }
   }

   bool has_err = false;

   // load init data from host to device
   for (int i = 0; i < total_matrixes; i++) {
       if (cudaMemcpy(mat_dev[i], mat_host[i], matrix_data_size, cudaMemcpyHostToDevice) != cudaSuccess) {
           printf("Failed to load initial data to device\n");
           has_err = true;
           break;
       }
   }

   if (!has_err) {
       // safe to lauch the kernel
       dim3 threadBlockDim(16, 16);
       dim3 threadGridDim(N/16, N/16);
       cuda_mat_add<<<threadGridDim, threadBlockDim, 0>>>(mat_dev[0], mat_dev[1], mat_dev[2]);

       if (cudaGetLastError() != cudaSuccess) {
           // failed to lauch kernel
           printf("Failed to launch cuda_mat_add kernel\n");
           has_err = true;
       }
   }

   if (!has_err) {
       // retrieve the result from device to host
       if (cudaMemcpy(mat_host[2], mat_dev[2], matrix_data_size, cudaMemcpyDeviceToHost) != cudaSuccess) {
           printf("Retrieving result failed\n");
           has_err = true;
       }
   }
   a = mat_host[0];
   b = mat_host[1];
   c = mat_host[2];
   if (!has_err) {
       for (int col = 0; col < N; col++) {
           for (int row = 0; row < N; row++) {
               float expected = *a++ + *b++;
               if (*c != expected) {
                   printf("Result is corrupted at (%d, %d) expected %f10.4f instead of %f10.4f\n",
                          row, col, expected, *c);
                   has_err = true;
                   break;
               }
               c++;
           }
           if (has_err) {
               break;
           }
       }
   }

   for (int i = 0; i < total_matrixes; i++) {
       free(mat_host[i]);
       cudaFree(mat_dev[i]);
   }

   if (has_err) {
       return EXIT_FAILURE;
   }
   printf("Success\n");
   return EXIT_SUCCESS;
}
