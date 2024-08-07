# AI HPC with Nvidia CUDA in C++
&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; -- Author: Bin Tan, [my linkedin](https://www.linkedin.com/in/bin-tan-3145721/)

This project provides in-depth tutorial articles and C++ example codes for AI HPC programming over Nvidia GPU devices.
Some performance characteritics have been discussed here. 

For the cluster level LLM performance, one may refer to my 
[LLM Cluster Performance](https://github.com/project-ai101/llm-cluster-perf) repository.

The following performance metric table demonstrates
a significant performance (total computation time) difference for solving the same matrix-matrix multiplication
problem in dimension (M=4096, N=4096, K=4096) with different hardware resource utilization and libraries usage.

|   Nvidia GPU   | GPU without L1 Cache Sharing | GPU Cores with L1 Cache Sharing | Tensor Cores with cuBLAS  | Tensor Cores with cuTENSOR |
|:--------------:|:----------------------------:|:-------------------------------:|:-------------------------:|:--------------------------:|
|      RTX 3060  |      733 milliseconds        |          164 milliseconds       |  29 milliseconds          |      24 milliseconds       |
| RTX 4060 Ti    |      432 milliseconds        |          117 milliseconds       |  18 milliseconds          |      17 milliseconds       |    
| RTX 4070 Ti S  |      233 milliseconds        |          59  milliseconds       |  14 milliseconds          |      23 milliseconds       |

The performance (total computation times including kernel launch) were measured over a GeForce RTX 3060 GPU card, a GeForce RTX 4060 Ti card
and a GeForce RTX 4070 Ti Super card.  RTX 3060 has Gen 3 Tensor cores and both RTX 4060 and RTX 4070 have Gen 4 Tensor cores.
The links to the implementation details are

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

### cuTENSOR
cuTENSOR is a Nvidia GPU accelerated library from Nvidia for tensor contraction, reduction and element-wise operations. Since a matrix
can be considered as a tensor with order 2, in this section, the matrix matrix multiplication is re-implemented in C++ with the cuTENSOR library.
The performance is about 17% improvement comparing with the implementation with the cuBLAS library.

- [Matrix-Matrix Multiplication in C++ with cuTENSOR](./cutensor/matrix-matrix-mulitply.md).

### Distributed GPUs with MPI and NCCL
For LLMs, training and inference may involve with many GPUs. A distributed computation framework is a must. 
MPI is a natural selection as it has been used for many years in HPC. Nvidia extends MPI to NCCL support multi-gpu computation. 

- [MPI Overview](./nccl/mpi_overview.md)
- [NCCL Overview](./nccl/nccl_overview.md)
- [Compute Matrix-Matrix Multiplication with two GPUs on the same host](./nccl/nccl_mmm_single_node.md)
- [Compute Matrix-Matrix Multiplication with four GPUs on two hosts](./nccl/nccl_mmm_multi_nodes.md)

### Nsight Compute
Nsight Compute is a performance analysis tool from Nvidia. For the old version Cuda, one may use the nvprof tool instead. 

### Nsight System
