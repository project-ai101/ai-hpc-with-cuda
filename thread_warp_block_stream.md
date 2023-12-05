# CUDA Thread, Warp, Block and Stream
  Author: Bin Tan

A NVidia GPU consists of an array of SM (Stream Multiprocessor). For example, a Nvidia RTX 3060 GPU has 28 SMs. 
Each SM has many CUDA cores which actually perform the instruction level computation. For example, a Nvidia RTX 3060 SM
has 128 CUDA cores. All these cores are scheduled by the Cuda Thread, Warp, Block and Stream framework. 

To increase the parallelism, a 32-threads form a thread warp. Each warp execute a single instruction at a time. Each thread
in a warp either executes the instruction or idles. A thread executing the instruction is called active thread. Such a design
allows each thread has its own execution path. In the chip level, threads are scheduled per warp basis. Further, all threads 
in a warp are located in a same thread block. 

Thread block is a logical concept and consists of a group of indexed threads. The threads in a block can be arranged into an 
one-dimensional, or two-dimensional, or three-dimensional block. The index of a thread in a block can be accessed in a GPU 
kernel code by (threadIdx.x, threadIdx.y, threadIdx.z). If the thread block is an one-dimensional block, both threadIdx.y and
threadIdx.z are zero. 

Further a group of thread blocks form a thread grid. A thread grid can be an one-dimentional, or two-dimensional, or
three-dimensional grid. Therefore, the index of a thread block in a grid is accessible in a GPU kernel code via (blockIdx.x,
blockIdx.y, blockIdx.z). As the same as the thread block, if the thread grid is an one-dimensional grid, both blockIdx.y and
blockIdx.z are zero.

How is the thread hirarchy related to a computation task? For example, let consider the following matrix addition,

```
     C = A + B, where A, B and C are M x N matrixes.
```

For the element pointwise, the matrix addition can be denoted as,

```
    c(i,j) = a(i, j) + b(i, j), where 0 <= i < M and 0 <= j < N.
```

To make explanation simple, let assume M = 4096 and N = 4096.  
And design M x N threads and layout them as the following two-dimensional lattice, 
```
               th th th th        ...      th th th th
               th th th th        ...      th th th th
                                  ...
               th th th th        ...      th th th th
```
With this design, each thread shall calculate one c(i, j) independently (aka complete parallelly). 
To make the threads managable, the lattice can be splitted
into many 16 x 16 blocks. Denote each block as TH. Now, we have 256 x 256 thread block grid,
```
               TH TH TH TH        ...      TH TH TH TH
               TH TH TH TH        ...      TH TH TH TH
                                  ...
               TH TH TH TH        ...      TH TH TH TH
where
                          th th th th th th th th th th th th th th th th
                          th th th th th th th th th th th th th th th th
                          th th th th th th th th th th th th th th th th
                          th th th th th th th th th th th th th th th th
                          th th th th th th th th th th th th th th th th
                          th th th th th th th th th th th th th th th th
                          th th th th th th th th th th th th th th th th
             TH   =       th th th th th th th th th th th th th th th th
                          th th th th th th th th th th th th th th th th
                          th th th th th th th th th th th th th th th th
                          th th th th th th th th th th th th th th th th
                          th th th th th th th th th th th th th th th th
                          th th th th th th th th th th th th th th th th
                          th th th th th th th th th th th th th th th th
                          th th th th th th th th th th th th th th th th
                          th th th th th th th th th th th th th th th th
```

From CUDA programming perspective, the computation task is implemented as a kernel 
function with \_\_global\_\_ prefix and the thread grid is defined by <<<...>>> syntax with type int or dim3.
One may ask why don't we just have a single giant 4096 x 4096 thread block. Unfortunetly, due to hardware 
resource limitation, each Stream Multiprocessor can support limited number of threads at the same time. Since
all threads in a thread block need to be allocated into a single SM at once, the total number of threads each block
can have is also limited, for example, 1024 threads per block. 

Now, let's have a runnable implementation of matrix addition. First, define the matrix addition kernel,

```cpp
__global__ void matrix_add(float* a, float* b, float* c) {
    // matrixes a, b, and c are in column-major
    // convert thread index into element location index
    int columnLen = gridDim.y * blockDim.y;
    int elementIdx = (blockIdx.x * blockDim.x + threadIdx.x) * columnLen +
                     (blockIdx.y * blockDim.y + threadIdx.y);
    c[elemntIdx] = a[elementIdx] + b[elementIdx];
}
```

Second, define the thread grid,
```cpp
    int N = 4096;
    ...
    dim3 threadBlockDim(16, 16);
    dim3 threadGridDim(N/16, N/16)
```

Third, link the computation task and the thread grid together via lauching the kernel.
```cpp
    float* a;
    float* b;
    float* c;
    ...
    matrix_add <<<threadGridDim, threadBlockDim>>>(a, b, c);
```

The final implementation, [cuda_matrix_add_example.cu](./cuda_matrix_add_example.cu).
