#include "test_operators.hpp"

template<typename T>
void PoolingOp<T>::PoolingForward(HipHandle& handle,
        PoolingDescriptor& poolSpec,
        const Tensor<T>& x, Tensor<T>& y){

    std::vector<int> workSpaceDims = {0};
    const T alpha = 1.0;
    const T beta = 0.0;
    
    miopenTensorDescriptor_t xDesc, yDesc;
    miopenPoolingDescriptor_t poolDesc;
    size_t workSpaceSize;
    
    CHECK_CALL_HIP(hipSetDevice(handle.deviceId()));
    CHECK_CALL_MIOPEN(miopenCreateTensorDescriptor(&xDesc));
    CHECK_CALL_MIOPEN(miopenCreateTensorDescriptor(&yDesc));
    
    CHECK_CALL_MIOPEN(miopenSet4dTensorDescriptor(xDesc, miopenFloat,
            x.dim(0), x.dim(1), x.dim(2), x.dim(3)));
    CHECK_CALL_MIOPEN(miopenSet4dTensorDescriptor(yDesc, miopenFloat,
            y.dim(0), y.dim(1), y.dim(2), y.dim(3)));
    
    CHECK_CALL_MIOPEN(miopenCreatePoolingDescriptor(&poolDesc));
    CHECK_CALL_MIOPEN(miopenSet2dPoolingDescriptor(poolDesc, poolSpec.getMode(),
            poolSpec.kernelshape[0], poolSpec.kernelshape[1],
            poolSpec.padding[0], poolSpec.padding[1],
            poolSpec.stride[0], poolSpec.stride[1]));
    
    CHECK_CALL_MIOPEN(miopenPoolingGetWorkSpaceSize(
            yDesc, &workSpaceSize));
    
    workSpaceDims[0] = static_cast<int>(workSpaceSize / sizeof(T));
    Tensor<T> workSpace(workSpaceDims);
    
    CHECK_CALL_MIOPEN(miopenPoolingForward(handle.miopenHandle(),
            poolDesc, &alpha, xDesc, x.data(),
            &beta, yDesc, y.data(),
            true, workSpace.data(), workSpaceSize));

    CHECK_CALL_MIOPEN(miopenDestroyPoolingDescriptor(poolDesc));
    CHECK_CALL_MIOPEN(miopenDestroyTensorDescriptor(xDesc));
    CHECK_CALL_MIOPEN(miopenDestroyTensorDescriptor(yDesc));
    handle.streamSynchronize();
}

template<typename T>
void PoolingOp<T>::PoolingBackward(HipHandle& handle,
        PoolingDescriptor& poolSpec,
        const Tensor<T>& x, const Tensor<T>& y,
        const Tensor<T>& dy, Tensor<T>& dx){
    
    std::vector<int> workSpaceDims = {0};
    const T alpha = 1.0;
    const T beta = 0.0;
    
    miopenTensorDescriptor_t xDesc, yDesc, dxDesc, dyDesc;
    miopenPoolingDescriptor_t poolDesc;
    size_t workSpaceSize;
    
    CHECK_CALL_HIP(hipSetDevice(handle.deviceId()));
    CHECK_CALL_MIOPEN(miopenCreateTensorDescriptor(&xDesc));
    CHECK_CALL_MIOPEN(miopenCreateTensorDescriptor(&yDesc));
    CHECK_CALL_MIOPEN(miopenCreateTensorDescriptor(&dxDesc));
    CHECK_CALL_MIOPEN(miopenCreateTensorDescriptor(&dyDesc));
    
    CHECK_CALL_MIOPEN(miopenSet4dTensorDescriptor(xDesc, miopenFloat,
            x.dim(0), x.dim(1), x.dim(2), x.dim(3)));
    CHECK_CALL_MIOPEN(miopenSet4dTensorDescriptor(yDesc, miopenFloat,
            y.dim(0), y.dim(1), y.dim(2), y.dim(3)));
    CHECK_CALL_MIOPEN(miopenSet4dTensorDescriptor(dxDesc, miopenFloat,
            dx.dim(0), dx.dim(1), dx.dim(2), dx.dim(3)));
    CHECK_CALL_MIOPEN(miopenSet4dTensorDescriptor(dyDesc, miopenFloat,
            dy.dim(0), dy.dim(1), dy.dim(2), dy.dim(3)));
    
    CHECK_CALL_MIOPEN(miopenCreatePoolingDescriptor(&poolDesc));
    CHECK_CALL_MIOPEN(miopenSet2dPoolingDescriptor(poolDesc, poolSpec.getMode(),
            poolSpec.kernelshape[0], poolSpec.kernelshape[1],
            poolSpec.padding[0], poolSpec.padding[1],
            poolSpec.stride[0], poolSpec.stride[1]));
    
    CHECK_CALL_MIOPEN(miopenPoolingGetWorkSpaceSize(
            yDesc, &workSpaceSize));
    
    workSpaceDims[0] = static_cast<int>(workSpaceSize / sizeof(T));
    Tensor<T> workSpace(workSpaceDims);
    
    CHECK_CALL_MIOPEN(miopenPoolingBackward(handle.miopenHandle(),
            poolDesc, &alpha, yDesc, y.data(),
            dyDesc, dy.data(), xDesc, x.data(),
            &beta, dxDesc, dx.data(), workSpace.data()));

    CHECK_CALL_MIOPEN(miopenDestroyPoolingDescriptor(poolDesc));
    CHECK_CALL_MIOPEN(miopenDestroyTensorDescriptor(xDesc));
    CHECK_CALL_MIOPEN(miopenDestroyTensorDescriptor(yDesc));
    CHECK_CALL_MIOPEN(miopenDestroyTensorDescriptor(dxDesc));
    CHECK_CALL_MIOPEN(miopenDestroyTensorDescriptor(dyDesc));
    handle.streamSynchronize();
}

template class PoolingOp<float>;
