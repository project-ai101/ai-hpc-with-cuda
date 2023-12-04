# AI HPC with Nvidia CUDA

This project provides in-depth tutorial articles and C++ example codes for AI HPC programming over Nvidia GPU devices.
Some performance characteritics have been discussed. 

### CUDA Concurrency
A GPU could have thousands of Cuda Cores. Each core can performance computation independently. To manage and schedule each core
is a non-trivial task. In Nvidia Cuda computation environment, thread, block, warp and stream form a complex scheculing system. 
This tutorial gives a bird view of the CUDA concurrency with a GPU Cuda core based matrix-matrix multiplication implementation.

- Cuda Thread, Warp, Block and Stream Overview
- A CUDA core based Matrix-Matrix multiplication example

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
NCCL is a distributed backend software to support multi-gpu computation. In this section, two examples are given.

- Compute Matrix-Matrix Multiplication with two GPUs on the same host.
- Compute Matrix-Matrix Multiplication with four GPUs on two hosts.

### Nsight Compute
Nsight Compute is a performance analysis tool from Nvidia. For the old version Cuda, one may use the nvprof tool instead. 

### Nsight System
