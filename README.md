# AI HPC with Nvidia CUDA

This project provides in-depth tutorial articles and example codes for AI HPC programming over Nvidia GPU devices. Performance is the main focus. 

### CUDA Thread and Block

### CUDA Stream

### cuBLAS
cuBLAS stands for Cuda Basic Linear Algebra Subroutines (BLAS). It is highly optimized BLAS APIs for Nvidia GPUs. 
It also leverages tensor cores for low and mix precision matrix multiplication. 
For the detailed reference documents, one may follow this [link](https://developer.nvidia.com/cublas) to Nvidia website.

Here, two simple examples are given to show how to use APIs in C++, some performance consideration are discussed
and stream based concurrency is reviewed. Hope they are helpful in utilize the benefits of the cuBLAS library.

- [A cuBLAS Matrix-Vector multiplication API example](./cublas_matrix_vector_multiplication_example.md) on how to use cuBLAS APIs in C++
- [A cuBLAS Matrix-Matrix multiplication API example](./cublas_matrix_matrix_multiplication_example.md) on how to use cuBLAS APIs in C++
- Performance characterisitics of cuBLAS
- Using cuBLAS with CUDA Stream for concurrency
### tensorRT

### NCCL

### Nsight Compute

### Nsight System
