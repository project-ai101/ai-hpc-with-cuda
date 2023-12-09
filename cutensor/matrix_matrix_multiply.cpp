/*******************************************************************************
 *
 *     Copyright (c) 2023 Bin Tan
 *
 *******************************************************************************/
#include <iostream>
#include "matrix_matrix_multiply.h"


MatrixTensor::MatrixTensor(int row_mode, int row_extend, int col_mode, int col_extend) {
    modes_[0] = row_mode;
    modes_[1] = col_mode;
    extends_[0] = row_extend;
    extends_[1] = col_extend;
    data_len_ = row_extend * col_extend * sizeof(float);
    host_mem_ = nullptr;
    dev_mem_ = nullptr;

    cu_data_type_ = CUTENSOR_R_32F;
}

MatrixTensor::~MatrixTensor() {
    if (host_mem_ != nullptr) {
        delete[] host_mem_;
    }

    if (dev_mem_ != nullptr) {
        cudaFree(dev_mem_);
    }
}

int MatrixTensor::init(cutensorHandle_t& handle) {
    handle_ = handle;
    host_mem_ = new float[extends_[0] * extends_[1]];
    if (cudaMalloc((void**) &dev_mem_, data_len_) != cudaSuccess) {
        std::cout << "Failed to allocate device memory" << std::endl;
        dev_mem_ = nullptr;
        return RETURN_FAILURE;
    }

    if( uintptr_t(dev_mem_) % DEVICE_MEM_ALIGNMENT != 0) {
        std::cout << "The allocated device memory is not well aligned" << std::endl;
        cudaFree(dev_mem_);
        dev_mem_ = nullptr;
        return RETURN_FAILURE;
    }

    if (cutensorCreateTensorDescriptor(handle_,
                                       &descriptor_,
                                       TOTAL_MODES,
                                       extends_,
                                       nullptr, /* stride */
                                       cu_data_type_,
                                       DEVICE_MEM_ALIGNMENT) != CUTENSOR_STATUS_SUCCESS) {
        std::cout << "Failed to create tensor descriptor" << std::endl;
        return RETURN_FAILURE;
    }
    return RETURN_SUCCESS;
}

int MatrixTensor::load_to_device() {
    if (cudaMemcpy(dev_mem_, host_mem_, data_len_, cudaMemcpyHostToDevice) != cudaSuccess) {
        std::cout << "Failed to load data into the device" << std::endl;
       return RETURN_FAILURE;
    }
    return RETURN_SUCCESS;
}

int MatrixTensor::retrieve_from_device() {
    if (cudaMemcpy(host_mem_, dev_mem_, data_len_, cudaMemcpyDeviceToHost) != cudaSuccess) {
        std::cout << "Failed to retrieve data from the device" << std::endl;
       return RETURN_FAILURE;
    }
    return RETURN_SUCCESS;
}

MatrixMatrixMultiply::MatrixMatrixMultiply(int m, int n, int k) {
    tensors_[A] = new MatrixTensor('i', m, 'k', k);
    tensors_[B] = new MatrixTensor('k', k, 'j', n);
    tensors_[C] = new MatrixTensor('i', m, 'j', n);

    comp_desc_ = CUTENSOR_COMPUTE_DESC_32F;
    algorithm_ = CUTENSOR_ALGO_DEFAULT;
    worksize_pref_ = CUTENSOR_WORKSPACE_DEFAULT;

    actual_workspace_ = nullptr;
    actual_workspace_size_ = 0;
    estimate_workspace_size_ = 0;
}

MatrixMatrixMultiply::~MatrixMatrixMultiply() {
    for (int i = 0; i < TOTAL_TENSORS; i++) {
        delete tensors_[i];
    }

    if (actual_workspace_ != nullptr) {
        cudaFree(actual_workspace_);
    }
}

int MatrixMatrixMultiply::init(cutensorHandle_t& handle) {
    handle_ = handle;

    for (int i = 0; i < TOTAL_TENSORS; i++) {
        if (tensors_[i]->init(handle_) != RETURN_SUCCESS) {
           std::cout << "Failed to initialize tensor " << i << std::endl;
           return RETURN_FAILURE;
        }
    }

    if (cutensorCreateContraction(handle_,
                                  &ops_desc_,
                                  tensors_[A]->getDescriptor(), tensors_[A]->getModes(), CUTENSOR_OP_IDENTITY,
                                  tensors_[B]->getDescriptor(), tensors_[B]->getModes(), CUTENSOR_OP_IDENTITY,
                                  tensors_[C]->getDescriptor(), tensors_[C]->getModes(), CUTENSOR_OP_IDENTITY,
                                  tensors_[C]->getDescriptor(), tensors_[C]->getModes(),
                                  comp_desc_) != CUTENSOR_STATUS_SUCCESS) {
        std::cout << "Failed to create contraction operation descriptor" << std::endl;
        return RETURN_FAILURE;
    }

    if (cutensorCreatePlanPreference(handle_,
                                     &plan_pref_,
                                     algorithm_,
                                     CUTENSOR_JIT_MODE_NONE) != CUTENSOR_STATUS_SUCCESS) {
        std::cout << "Failed to create plan preference" << std::endl;
        return RETURN_FAILURE;
    }

    if (cutensorEstimateWorkspaceSize(handle_,
                                      ops_desc_,
                                      plan_pref_,
                                      worksize_pref_,
                                      &estimate_workspace_size_) != CUTENSOR_STATUS_SUCCESS) {
        std::cout << "Failed to query estimate workspace size" << std::endl;
        return RETURN_FAILURE;
    }

    if (cutensorCreatePlan(handle_,
                           &plan_,
                           ops_desc_,
                           plan_pref_,
                           estimate_workspace_size_) != CUTENSOR_STATUS_SUCCESS) {
        std::cout << "Failed to create constraction plan" << std::endl;
        return RETURN_FAILURE;
    }

    if (cutensorPlanGetAttribute(handle_,
                                 plan_,
                                 CUTENSOR_PLAN_REQUIRED_WORKSPACE,
                                 &actual_workspace_size_,
                                 sizeof(actual_workspace_size_)) != CUTENSOR_STATUS_SUCCESS) {
        std::cout << "Failed to query the actual workspace size associated with the contraction plan" << std::endl;
        return RETURN_FAILURE;
    }

    if (actual_workspace_size_ > estimate_workspace_size_) {
        std::cout << "Actual workspace size is greater than estimated" << std::endl;
        return RETURN_FAILURE;
    }

    if (actual_workspace_size_ > 0) {
        if (cudaMalloc(&actual_workspace_, actual_workspace_size_) != cudaSuccess) {
            actual_workspace_ = nullptr;
            std::cout << "Failed to allocate acutal workspace" << std::endl;
            return RETURN_FAILURE;
        }
    }

    return RETURN_SUCCESS;
}

int MatrixMatrixMultiply::load_data() {
    for (int i = 0; i < TOTAL_TENSORS; i++) {
         if (tensors_[i] -> load_to_device() != RETURN_SUCCESS) {
             std::cout << "Failed to load data into device" << std::endl;
             return RETURN_FAILURE;
         }
    }
    return RETURN_SUCCESS;
}

int MatrixMatrixMultiply::perform_comp(float alpha, float beta) {
    if (cutensorContract(handle_,
                         plan_,
                         (void*) &alpha, tensors_[A]->getDeviceMem(), tensors_[B]->getDeviceMem(),
                         (void*) &beta, tensors_[C]->getDeviceMem(), tensors_[C]->getDeviceMem(),
                         actual_workspace_, actual_workspace_size_, 0) != CUTENSOR_STATUS_SUCCESS) {
        std::cout << "Failed to perform tensor contraction" << std::endl;
        return RETURN_FAILURE;
    }
    return RETURN_SUCCESS;
}


int MatrixMatrixMultiply::retrieve_result() {
    if (tensors_[C] -> retrieve_from_device() != RETURN_SUCCESS) {
        std::cout << "Failed to retrieve result from device" << std::endl;
        return RETURN_FAILURE;
    }
    return RETURN_SUCCESS;
}
                                                                                                                                                                                                                                                               
