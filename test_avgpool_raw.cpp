#include "test_helper.hpp"
    
struct PoolingDescriptor {
    int pooldim = 2;
    std::string mode;
    std::vector<int> kernelshape;
    std::vector<int> padding;
    std::vector<int> stride;
    PoolingDescriptor(std::string mode_in,
            int kernel_h, int kernel_w, 
            int padding_h, int padding_w,
            int stride_h, int stride_w){
        mode = mode_in;
        kernelshape.push_back(kernel_h);
        kernelshape.push_back(kernel_w);
        padding.push_back(padding_h);
        padding.push_back(padding_w);
        stride.push_back(stride_h);
        stride.push_back(stride_w);
    }
    miopenPoolingMode_t getMode(){
        if(mode == "avg") {
            return miopenPoolingAverage;
        } else if(mode == "max") {
            return miopenPoolingMax;
        } else {
            std::cerr << "Error: Unknown pooling mode!" << std::endl;
            exit(1);
        }
    }
};

template<typename T>
void testPoolingForward(HipHandle& handle,
        PoolingDescriptor& poolSpec,
        const std::vector<T>& x, std::vector<int> xSpec,
        const std::vector<T>& y, std::vector<int> ySpec){
    
    Tensor<T> xData(x, xSpec);
    Tensor<T> yData(ySpec);
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
            xSpec[0], xSpec[1], xSpec[2], xSpec[3]));
    CHECK_CALL_MIOPEN(miopenSet4dTensorDescriptor(yDesc, miopenFloat,
            ySpec[0], ySpec[1], ySpec[2], ySpec[3]));
    
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
            poolDesc, &alpha, xDesc, xData.data(),
            &beta, yDesc, yData.data(),
            true, workSpace.data(), workSpaceSize));

    CHECK_CALL_MIOPEN(miopenDestroyPoolingDescriptor(poolDesc));
    CHECK_CALL_MIOPEN(miopenDestroyTensorDescriptor(xDesc));
    CHECK_CALL_MIOPEN(miopenDestroyTensorDescriptor(yDesc));
    handle.streamSynchronize();
    
    testSame(yData, y, std::string("Forward output-yData"));
}

template<typename T>
void testPoolingBackward(HipHandle& handle,
        PoolingDescriptor& poolSpec,
        const std::vector<T>& x, std::vector<int> xSpec,
        const std::vector<T>& y, std::vector<int> ySpec,
        const std::vector<T>& dy, std::vector<int> dySpec,
        const std::vector<T>& dx, std::vector<int> dxSpec){
    
    Tensor<T> xData(x, xSpec);
    Tensor<T> yData(y, xSpec);
    Tensor<T> dyData(dyData, dySpec);
    Tensor<T> dxData(dxSpec);
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
            xSpec[0], xSpec[1], xSpec[2], xSpec[3]));
    CHECK_CALL_MIOPEN(miopenSet4dTensorDescriptor(yDesc, miopenFloat,
            ySpec[0], ySpec[1], ySpec[2], ySpec[3]));
    CHECK_CALL_MIOPEN(miopenSet4dTensorDescriptor(dxDesc, miopenFloat,
            dxSpec[0], dxSpec[1], dxSpec[2], dxSpec[3]));
    CHECK_CALL_MIOPEN(miopenSet4dTensorDescriptor(dyDesc, miopenFloat,
            dySpec[0], dySpec[1], dySpec[2], dySpec[3]));
    
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
            poolDesc, &alpha, yDesc, yData.data(),
            dyDesc, dyData.data(), xDesc, xData.data(),
            &beta, dxDesc, dxData.data(), workSpace.data()));

    CHECK_CALL_MIOPEN(miopenDestroyPoolingDescriptor(poolDesc));
    CHECK_CALL_MIOPEN(miopenDestroyTensorDescriptor(xDesc));
    CHECK_CALL_MIOPEN(miopenDestroyTensorDescriptor(yDesc));
    CHECK_CALL_MIOPEN(miopenDestroyTensorDescriptor(dxDesc));
    CHECK_CALL_MIOPEN(miopenDestroyTensorDescriptor(dyDesc));
    handle.streamSynchronize();
    
    testSame(dxData, dx, std::string("Backward output-dxData"));
}

int main(int argc, char** argv) {

    CHECK_CALL_HIP(hipFree(0));
    
    std::vector<float> x_std {
        2, 2, 2, 2,
        2, 1, 1, 2,
        2, 1, 1, 2,
        2, 2, 2, 2
    };
    std::vector<int> x_shape {1, 1, 4, 4};

    std::vector<float> y_std_avg_includepad {
        0.5, 1, 0.5,
        1, 1, 1,
        0.5, 1, 0.5
    };
    std::vector<int> y_shape_avg_includepad {1, 1, 3, 3};

    std::vector<float> y_std_avg_notincludepad {
        2, 2, 2,
        2, 1, 2,
        2, 2, 2
    };
    std::vector<int> y_shape_avg_notincludepad {1, 1, 3, 3};

    std::vector<float> y_std_avg_nopad{
        1.75, 1.75,
        1.75, 1.75
    };
    std::vector<int> y_shape_avg_nopad {1, 1, 2, 2};

    PoolingDescriptor avgPoolDescPadding("avg", 2, 2, 1, 1, 2, 2);
    PoolingDescriptor avgPoolDescNoPadding("avg", 2, 2, 0, 0, 2, 2);
    
    PoolingDescriptor maxPoolDescPadding("max", 2, 2, 1, 1, 2, 2);
    PoolingDescriptor maxPoolDescNoPadding("max", 2, 2, 0, 0, 2, 2);
    
    HipHandle handle(0);
    std::cout << "Start avgpooling test with include padding" << std::endl;
    testPoolingForward(handle, avgPoolDescPadding,
            x_std, x_shape,
            y_std_avg_includepad, y_shape_avg_includepad);

    std::cout << "Start avgpooling test without include padding" << std::endl;
    testPoolingForward(handle, avgPoolDescPadding,
            x_std, x_shape,
            y_std_avg_notincludepad, y_shape_avg_notincludepad);

    std::cout << "Start avgpooling test without padding" << std::endl;
    testPoolingForward(handle, avgPoolDescNoPadding,
            x_std, x_shape,
            y_std_avg_nopad, y_shape_avg_nopad);
    return 0;
}
