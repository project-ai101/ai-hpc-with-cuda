/*******************************************************************************
 *
 *     Copyright (c) 2023 Bin Tan
 *
 *******************************************************************************/

template<class ValType> __global__ void cuda_mat_add(ValType* a, ValType* b, ValType *c) {
   int element_x = blockIdx.x * blockDim.x + threadIdx.x;
   int element_y = blockIdx.y * blockDim.y + threadIdx.y;
   int column_len = gridDim.y * blockDim.y;

   int element_id = element_x * column_len + element_y;
   c[element_id] = a[element_id] + b[element_id];
}


template<class ValType> CudaMatMatAddComp<ValType>::CudaMatMatAddComp(int m, int n) {
    m_ = m;
    n_ = n;
    matrix_size_ = m_ * n_;
    matrix_data_size_ = matrix_size_ * sizeof(ValType);

    for (int i = 0; i < TOTAL_MATRIXES; i++) {
        mat_host_[i] = nullptr;
        mat_dev_[i]  = nullptr;
    }
}

template<class ValType> CudaMatMatAddComp<ValType>::~CudaMatMatAddComp() {
    for (int i = 0; i < TOTAL_MATRIXES; i++) {
        if (mat_host_[i] != nullptr) {
            delete[] mat_host_[i];
        }
        if (mat_dev_[i]  != nullptr) {
            cudaFree(mat_dev_[i]);
        }
    }
}

template<class ValType> int CudaMatMatAddComp<ValType>::init_mem() {
   if (m_ % BLOCK_DIM_Y != 0 || n_ % BLOCK_DIM_X != 0) {
       std::cout << "the matrix dimension (" << m_ << ", " << n_ << ") "
                 << "must be divided by the block dimension ("
                 << BLOCK_DIM_Y << ", " << BLOCK_DIM_X << ")"
                 << std::endl;
       return RETURN_ERROR;
   }

   for (int i = 0; i < TOTAL_MATRIXES; i++) {
        if (cudaMalloc((void**)&mat_dev_[i], matrix_data_size_) != cudaSuccess) {
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
        mat_host_[i] = new float[matrix_size_];
   }

   return RETURN_SUCCESS;
}

template<class ValType> int CudaMatMatAddComp<ValType>::load_data() {
   for (int i = 0; i < TOTAL_MATRIXES; i++) {
       if (cudaMemcpy(mat_dev_[i], mat_host_[i], matrix_data_size_, cudaMemcpyHostToDevice) != cudaSuccess) {
           std::cout << "Failed to load initial data to device" << std::endl;
           return RETURN_ERROR;
       }
   }

   return RETURN_SUCCESS;
}

template<class ValType> int CudaMatMatAddComp<ValType>::perform_comp() {
    dim3 threadBlockDim(BLOCK_DIM_Y, BLOCK_DIM_X);
    dim3 threadGridDim(m_/BLOCK_DIM_Y, n_/BLOCK_DIM_X);
    cuda_mat_add<<<threadGridDim, threadBlockDim, 0>>>(mat_dev_[0], mat_dev_[1], mat_dev_[2]);

    if (cudaGetLastError() != cudaSuccess) {
        // failed to lauch kernel
        std::cout << "Failed to launch cuda_mat_add kernel" << std::endl;
        return RETURN_ERROR;
    }

    return RETURN_SUCCESS;
}

template<class ValType> int CudaMatMatAddComp<ValType>::retrieve_result() {
    if (cudaMemcpy(mat_host_[C], mat_dev_[C], matrix_data_size_, cudaMemcpyDeviceToHost) != cudaSuccess) {
        printf("Retrieving result failed\n");
        return RETURN_ERROR;
    }
    return RETURN_SUCCESS;
}
