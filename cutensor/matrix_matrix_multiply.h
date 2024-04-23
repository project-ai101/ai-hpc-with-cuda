/*******************************************************************************
 *
 *     Copyright (c) 2023 Bin Tan
 *
 *******************************************************************************/
ifndef CUTENSOR_MATRIX_MATRIX_MULTIPLY_HH_
#define CUTENSOR_MATRIX_MATRIX_MULTIPLY_HH_

#include <cuda_runtime.h>
#include <cutensor.h>
#include <cutensor/types.h>

#define RETURN_FAILURE -1
#define RETURN_SUCCESS 1

#define DEVICE_MEM_ALIGNMENT 128

class MatrixMatrixMultiply;

class MatrixTensor {
    protected:
        enum ModeIndex {
            ROW = 0,
            COL,
            TOTAL_MODES
        };
        cutensorHandle_t handle_;
        cutensorDataType_t cu_data_type_;
        cutensorTensorDescriptor_t descriptor_;

        int modes_[TOTAL_MODES];
        long extends_[TOTAL_MODES];
        unsigned int data_len_;                    // in bytes
        float* host_mem_;
        float* dev_mem_;

        // allow MatrixMatrixMultiply function object to access these APIs
        friend class MatrixMatrixMultiply;
        int load_to_device();
        int retrieve_from_device();
        float* getDeviceMem() { return dev_mem_; }

        MatrixTensor(int row_mode, int row_extend, int col_mode, int col_extend);
        int init(cutensorHandle_t& handle);
    public:
        virtual ~MatrixTensor();
        // column-major
        unsigned int getRows() { return extends_[ROW]; }
        unsigned int getCols() { return extends_[COL]; }
        const int* getModes() { return modes_; }
        const long* getExtends() { return extends_; }

        cutensorDataType_t getDataType() { return cu_data_type_; }
        cutensorTensorDescriptor_t getDescriptor() { return descriptor_; }
        float* getHostMem() { return host_mem_; }
};

class MatrixMatrixMultiply {
    protected:
        enum TensorIndex {
            A = 0,
            B,
            C,
            TOTAL_TENSORS
        };

        cutensorHandle_t handle_;
        MatrixTensor* tensors_[TOTAL_TENSORS];
        cutensorOperationDescriptor_t ops_desc_;
        cutensorComputeDescriptor_t comp_desc_;
        cutensorAlgo_t algorithm_;
        cutensorWorksizePreference_t worksize_pref_;
        void* actual_workspace_;
        unsigned long actual_workspace_size_;
        unsigned long estimate_workspace_size_;

        cutensorPlanPreference_t plan_pref_;
        cutensorPlan_t plan_;
    public:
        MatrixMatrixMultiply(int m, int n, int k);
        virtual ~MatrixMatrixMultiply();

        int init(cutensorHandle_t& handle);
        int load_data();
        int perform_comp(float alpha, float beta);
        int retrieve_result();

        MatrixTensor* getA() { return tensors_[A]; }
        MatrixTensor* getB() { return tensors_[B]; }
        MatrixTensor* getC() { return tensors_[C]; }
};

#endif
