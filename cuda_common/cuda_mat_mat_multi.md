# CUDA Matrix-Matrix Multiplication in C++
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; -- Author: Bin Tan

### Overview
The computation task is to calculate the matrix-matrix multiplication.

```math
\left(1\right)\hspace{9cm}        C = alpha * A * B + beta * C \hspace{8cm}
```
where A is a M x K matrix, B is a K x N matrix and C is a M x N matrix.
Both alpha and beta are scalar float values.

From element-wise, the problem can be interpreted as 

```math
\left(2\right)\hspace{4cm}       C(i, j) = alpha * \sum_{l=0}^{K} A(i, l) * B(l, j) + beta * C(i, j), \quad
where \quad 0 <= i < M,  \quad 0 <= j < N
```

Above element-wise interpretation gives a nature task partition. Computation of 
each $` C(i, j) `$ forms an independent sub task. And, all sub tasks can be peroformed
parallelly. Therefore, we can assign one CUDA thread to each one $` C(i, j) `$ sub task.

Though the implementation of above approach is fairly straightforward, there are some
performance drawbacks. For example, for the sub tasks $` C(i, j_{1}) and C(i, j_{2}) `$, both of
them need to load K elements of $` A(i, l) `$, where $` 0 <= l < K `$, into registers from main memory
independently. Denote $`\left(2\right)`$ as the Slow_Path which is implemented by 
[cuda_mat_mat_multiply_slow](./cuda_mat_mat_multiply.cpp).

To utilize the small L1 cache, we can parition the C into many small sub-matrixes $` C_{i, j} `$, 
for example, 16 x 16 sub-matrixes.

```math
C = \begin{pmatrix}
     C_{0,0} & C_{0,1} & \cdots & C_{0, n-1} \\
     C_{1,0} & C_{1,1} & \cdots & C_{1, n-1} \\ 
     \vdots  & \vdots  & \ddots & \vdots      \\
     C_{m-1,0} & C_{m-1,1} & \cdots & C_{m-1, n-1}
    \end{pmatrix}
```
where $` m = M/16, n = N/16`$.
Then, correspondently, partition both A and B into the same 16x16 sub-matrixes, $` A_{i, j} `$ and $` B_{i, j} `$. 
```math
A = \begin{pmatrix}
     A_{0,0} & A_{0,1} & \cdots & A_{0, k-1} \\
     A_{1,0} & A_{1,1} & \cdots & A_{1, k-1} \\ 
     \vdots  & \vdots  & \ddots & \vdots      \\
     A_{m-1,0} & A_{m-1,1} & \cdots & A_{m-1, k-1}
    \end{pmatrix}
\hspace{4cm}
B = \begin{pmatrix}
     B_{0,0} & B_{0,1} & \cdots & B_{0, n-1} \\
     B_{1,0} & B_{1,1} & \cdots & B_{1, n-1} \\ 
     \vdots  & \vdots  & \ddots & \vdots      \\
     B_{k-1,0} & B_{k-1,1} & \cdots & B_{k-1, n-1}
    \end{pmatrix}
```
where $` m = M/16, n = N/16, k = K/16`$. Then, each 16 x 16 sub-matrix $` C_{i, j} `$ can be computed with
```math
\left( 3 \right) \hspace{5cm} C_{i, j} = alpha * \sum_{l=0}^{k} A_{i, l} * B_{l, j} + beta * C_{i, j}, \quad
where \quad 0 <= i < m,  \quad 0 <= j < n
```
With the same layout of $` C_{i,j} `$, a 16 x 16 CUDA thread block can be designed to compute $` C_{i, j} `$. 
Further, both $` A_{i, l} `$ and $` B_{l, j} `$ are small enough to be loaded into L1 cache for shared by
all threads in a thread block. This shall substantially increase the performance. $` \left( 3 \right) `$ is called
as the Fast_Path which is implemented by [cuda_mat_mat_multiply_fast](./cuda_mat_mat_multiply.cpp).

### C++ Files

The complete implementation of both Slow_Path and Fast_Path of Matrix-Matrix multiplication can be found in the following
source file.

- [cuda_mat_mat_multiply.h](./cuda_mat_mat_multiply.h)
- [cuda_mat_mat_multiply.cpp](./cuda_mat_mat_multiply.cpp)
- [cuda_mat_mat_multiply_example.cu](./cuda_mat_mat_multiply_example.cu)

Use following command to compile them,

```
$ nvcc cuda_mat_mat_multiply_example.cu -o cuda_mat_mat_multiply
```

To run it with fast path,
```
$ ./cuda_mat_mat_multiply
Success
Matrix-Matrix-Multiplication - Fast Path:  size (4096, 4096, 4096), total comp time 164.55 milliseconds
```

To run it with slow path,
```
$ ./cuda_mat_mat_multiply false
Success
Matrix-Matrix-Multiplication - Slow Path:  size (4096, 4096, 4096), total comp time 733.452 milliseconds
```
