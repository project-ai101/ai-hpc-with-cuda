/*******************************************************************************
 *
 *     Copyright (c) 2023 Bin Tan
 *
 *******************************************************************************/

template<class ValType> __global__ void cuda_mat_multiply_slow(int m, int n, int k, ValType* a, ValType* b, ValType *c, float alpha, float beta) {
    // matrix a is a row-major matrix
    ValType* a_row_start = a + (blockIdx.y * blockDim.y + threadIdx.y) * k;
    // matrix b is a column-major matrix
    ValType* b_col_start = b + (blockIdx.x * blockDim.x + threadIdx.x) * k;

    float val = 0.0f;
    for (int i = 0; i < k; i++) {
        val += (*a_row_start++) * (*b_col_start++);
    }

   // matrix C is a column-major matrix
   int c_pos = (blockIdx.x * blockDim.x + threadIdx.x) * m + blockIdx.y * blockDim.y + threadIdx.y;
   c[c_pos] = alpha * val + beta * c[c_pos];
}

template<class ValType> __global__ void cuda_mat_multiply_fast(int m, int n, int k, ValType* a, ValType* b, ValType *c, float alpha, float beta) {
   // matrix A is a row-major matrix
   int a_sub_data_start_pos = blockIdx.y * blockDim.y * k;
   // matrix B is a column-major matrix
   int b_sub_data_start_pos = blockIdx.x * blockDim.x * k;

   __shared__ ValType a_sub_mat[BLOCK_SIZE][BLOCK_SIZE];
   __shared__ ValType b_sub_mat[BLOCK_SIZE][BLOCK_SIZE];

   float val = 0.0f;

   int total_k_steps = k / BLOCK_SIZE;

   int a_sub_data_local_pos = threadIdx.y * k + threadIdx.x;
   int b_sub_data_local_pos = threadIdx.x * k + threadIdx.y;
   for (int k_step = 0; k_step < total_k_steps; k_step++) {
       a_sub_mat[threadIdx.y][threadIdx.x] = a[a_sub_data_start_pos + a_sub_data_local_pos];
       b_sub_mat[threadIdx.y][threadIdx.x] = b[b_sub_data_start_pos + b_sub_data_local_pos];

       __syncthreads();

       for (int k_idx = 0; k_idx < BLOCK_SIZE; k_idx++) {
           val += a_sub_mat[threadIdx.y][k_idx] * b_sub_mat[k_idx][threadIdx.x];
       }

       __syncthreads();
       a_sub_data_start_pos += BLOCK_SIZE;
       b_sub_data_start_pos += BLOCK_SIZE;
   }


   // matrix C is a column-major matrix
   int c_pos = (blockIdx.x * blockDim.x + threadIdx.x) * m + blockIdx.y * blockDim.y + threadIdx.y;
   c[c_pos] = alpha * val + beta * c[c_pos];
}


template<class ValType> CudaMatMatMultiplyComp<ValType>::CudaMatMatMultiplyComp(int m, int n, int k, float alpha, float beta) {
    m_ = m;
    n_ = n;
    k_ = k;
    matrix_size_[A] = m_ * k_;
    matrix_size_[B] = k_ * n_;
    matrix_size_[C] = m_ * n_;

    alpha_ = alpha;
    beta_ = beta;

    for (int i = 0; i < TOTAL_MATRIXES; i++) {
        matrix_data_size_[i] = matrix_size_[i] * sizeof(ValType);
    }

    for (int i = 0; i < TOTAL_MATRIXES; i++) {
        mat_host_[i] = nullptr;
        mat_dev_[i]  = nullptr;
    }
}

template<class ValType> CudaMatMatMultiplyComp<ValType>::~CudaMatMatMultiplyComp() {
    for (int i = 0; i < TOTAL_MATRIXES; i++) {
        if (mat_host_[i] != nullptr) {
            delete[] mat_host_[i];
        }
        if (mat_dev_[i]  != nullptr) {
            cudaFree(mat_dev_[i]);
        }
    }
}

template<class ValType> int CudaMatMatMultiplyComp<ValType>::init_mem() {
   if (m_ % BLOCK_SIZE != 0 || n_ % BLOCK_SIZE != 0 || k_ % BLOCK_SIZE != 0) {
       std::cout << "the matrix dimension (" << m_ << ", " << n_ << ", " << k_ << ") "
                 << "must be divided by the block dimension ("
                 << BLOCK_SIZE << ", " << BLOCK_SIZE << "," << BLOCK_SIZE << ")"
                 << std::endl;
       return RETURN_ERROR;
   }

   for (int i = 0; i < TOTAL_MATRIXES; i++) {
        if (cudaMalloc((void**)&mat_dev_[i], matrix_data_size_[i]) != cudaSuccess) {
            for (int j = 0; j < TOTAL_MATRIXES; j++) {
                 if (mat_dev_[j] != nullptr) {
                     cudaFree(mat_dev_[j]);
                 }
            }
            std::cout << "Failed to allocated device memory for matrix " << i << std::endl;
            return RETURN_ERROR;
        }
   }


   // allocate host side matrix memory
   for (int i = 0; i < TOTAL_MATRIXES; i++) {
        mat_host_[i] = new float[matrix_size_[i]];
   }

   return RETURN_SUCCESS;
}

template<class ValType> int CudaMatMatMultiplyComp<ValType>::load_data() {
   for (int i = 0; i < TOTAL_MATRIXES; i++) {
       if (cudaMemcpy(mat_dev_[i], mat_host_[i], matrix_data_size_[i], cudaMemcpyHostToDevice) != cudaSuccess) {
           std::cout << "Failed to load initial data to device" << std::endl;
           return RETURN_ERROR;
       }
   }

   return RETURN_SUCCESS;
}

template<class ValType> int CudaMatMatMultiplyComp<ValType>::perform_comp(bool use_fast_path) {
    dim3 threadBlockDim(BLOCK_SIZE, BLOCK_SIZE);
    dim3 threadGridDim(m_/BLOCK_SIZE, n_/BLOCK_SIZE);

    if (use_fast_path) {
        cuda_mat_multiply_fast<<<threadGridDim, threadBlockDim, 0>>>(m_, n_, k_, mat_dev_[0], mat_dev_[1], mat_dev_[2], alpha_, beta_);
    } else {
        cuda_mat_multiply_slow<<<threadGridDim, threadBlockDim, 0>>>(m_, n_, k_, mat_dev_[0], mat_dev_[1], mat_dev_[2], alpha_, beta_);
    }
    if (cudaGetLastError() != cudaSuccess) {
        // failed to lauch kernel
        std::cout << "Failed to launch cuda_mat_multiply kernel" << std::endl;
        return RETURN_ERROR;
    }

    return RETURN_SUCCESS;
}

template<class ValType> int CudaMatMatMultiplyComp<ValType>::retrieve_result() {
    if (cudaMemcpy(mat_host_[C], mat_dev_[C], matrix_data_size_[C], cudaMemcpyDeviceToHost) != cudaSuccess) {
        printf("Retrieving result failed\n");
        return RETURN_ERROR;
    }
    return RETURN_SUCCESS;
}
