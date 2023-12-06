# AI HPC with Nvidia CUDA in C++
&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; -- Author: Bin Tan

This project provides in-depth tutorial articles and C++ example codes for AI HPC programming over Nvidia GPU devices.
Some performance characteritics have been discussed. For example, the following performance metric table demonstrates
a significant performance (total computation time) difference for solving the same matrix-matrix multiplication
problem with different hardware resource utilization.

| GPU without L1 Cache Sharing |   GPU with L1 Cache Sharing   |     Tensor Cores with cuBLAS        |
|:----------------------------:|:-----------------------------:|:-----------------------------------:|
|       733 milliseconds       |          164 milliseconds     |          29 milliseconds            |

The performance (total computation times) were measured over a GeForce RTX 3060 GPU card. The links to the implementation
details are

- [GPU without L1 Cache Sharing](./cuda_common/cuda_mat_mat_multi.md) (aka use_slow_path)
- [GPU with L1 Cache Sharing](./cuda_common//cuda_mat_mat_multi.md) (aka use_fast_path)
- [Tensor Cores with cuBLAS](./cublas/cublas_matrix_matrix_multiplication_example.md)

### CUDA Concurrency
A GPU could have thousands of Cuda Cores. Each core can performance computation independently. To manage and schedule each core
is a non-trivial task. In Nvidia Cuda computation environment, thread, block, warp and stream form a complex scheculing system. 
This tutorial gives a bird view of the CUDA concurrency with a GPU Cuda core based matrix-matrix multiplication implementation.

- [Cuda Thread, Warp, Block and Stream Overview](./cuda_common/thread_warp_block_stream.md)
- [A Cuda core based Matrix-Matrix addition in C++](./cuda_common/cuda_mat_mat_add.md)
- [A Cuda core based Matrix-Matrix multiplication in C++](./cuda_common/cuda_mat_mat_multi.md)

### cuBLAS
cuBLAS stands for Cuda Basic Linear Algebra Subroutines (BLAS). It is highly optimized BLAS APIs for Nvidia GPUs. 
It also leverages tensor cores for low and mix precision matrix multiplication. 
For the detailed reference documents, one may follow this [link](https://developer.nvidia.com/cublas) to Nvidia website.

Here, two simple examples are given to show how to use APIs in C++, some performance consideration are discussed
and stream based concurrency is reviewed. Hope they are helpful in utilize the benefits of the cuBLAS library.

- [A cuBLAS Matrix-Vector multiplication API example](./cublas/cublas_matrix_vector_multiplication_example.md) on how to use cuBLAS APIs in C++
- [A cuBLAS Matrix-Matrix multiplication API example](./cublas/cublas_matrix_matrix_multiplication_example.md) on how to use cuBLAS APIs in C++
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
