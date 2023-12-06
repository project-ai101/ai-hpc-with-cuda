# CUDA Matrix-Matrix Multiplication in C++
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; -- Author: Bin Tan

### Overview
The computation task is to calculate the matrix-matrix multiplication.

```math
\left(1\right)\hspace{3cm}        C = alpha * A * B + beta * C
```
where A is a M x K matrix, B is a K x N matrix and C is a M x N matrix.
Both alpha and beta are scalar float values.

From element-wise, the problem can be interpreted as 

```math
\left(2\right)\hspace{3cm}       C(i, j) = alpha * \sum_{l=0}^{K} A(i, l) * B(l, j) + beta * C(i, j), \quad
where \quad 0 <= i < M,  \quad 0 <= j < N
```

Above element-wise interpretation gives a nature task partition. Computation of 
each C(i, j) forms an independent sub task. And, all sub tasks can be peroformed
parallelly. Therefore, we can assign one CUDA thread to each one C(i, j) sub task.

Though the implementation of above approach is fairly straightforward, there are some
performance drawbacks. For example, for the sub tasks C(i, j1) and C(i, j2), both of
them need to load K elements of A(i, l), where 0 <= l < K, into registers from main memory
independently. Denote $`\left(2\right)`$ as the Slow_Path.



### C++ Files

- [cuda_mat_mat_multiply.h](./cuda_mat_mat_multiply.h)
- [cuda_mat_mat_multiply.cpp](./cuda_mat_mat_multiply.cpp)
- [cuda_mat_mat_multiply_example.cu](./cuda_mat_mat_multiply_example.cu)

  
