#include "test_operators.hpp"

// Convolution Ops
template<typename T>
void ConvolutionOp<T>::ConvForward(HipHandle& handle,
        ConvDescriptor& convSpec,
        const Tensor<T>& x, const Tensor<T>& w,
        const Tensor<T>* bias, Tensor<T>& y){

    if (convSpec.mode == "conv") {
        if (convSpec.dilation.size() != 0) {
            CHECK_ARGS(convSpec.dilation[0] == 1 &&
                    convSpec.dilation[1] == 1,
                    "Invalid dilation for convolution!");
        }
        convSpec.dilation.push_back(1);
        convSpec.dilation.push_back(1);
    }

    std::vector<int> workSpaceDims = {0};
    const T alpha = 1.0;
    const T beta = 0.0;
    
    miopenTensorDescriptor_t xDesc, wDesc, yDesc;
    miopenConvolutionDescriptor_t convDesc;
    int returnedAlgoCount;
    miopenConvAlgoPerf_t perfResults;
    size_t workSpaceSize;
    
    CHECK_CALL_HIP(hipSetDevice(handle.deviceId()));
    CHECK_CALL_MIOPEN(miopenCreateTensorDescriptor(&xDesc));
    CHECK_CALL_MIOPEN(miopenCreateTensorDescriptor(&wDesc));
    CHECK_CALL_MIOPEN(miopenCreateTensorDescriptor(&yDesc));
    
    CHECK_CALL_MIOPEN(miopenSet4dTensorDescriptor(xDesc, miopenFloat,
            x.dim(0), x.dim(1), x.dim(2), x.dim(3)));
    CHECK_CALL_MIOPEN(miopenSet4dTensorDescriptor(wDesc, miopenFloat,
            w.dim(0), w.dim(1), w.dim(2), w.dim(3)));
    CHECK_CALL_MIOPEN(miopenSet4dTensorDescriptor(yDesc, miopenFloat,
            y.dim(0), y.dim(1), y.dim(2), y.dim(3)));
    CHECK_CALL_MIOPEN(miopenCreateConvolutionDescriptor(&convDesc));
    CHECK_CALL_MIOPEN(miopenInitConvolutionDescriptor(convDesc,
            convSpec.getMode(),
            convSpec.padding[0], convSpec.padding[1],
            convSpec.stride[0], convSpec.stride[1],
            convSpec.dilation[0], convSpec.dilation[1]));
    
    CHECK_CALL_MIOPEN(miopenConvolutionForwardGetWorkSpaceSize(
            handle.miopenHandle(),
            wDesc, xDesc, convDesc, yDesc,
            &workSpaceSize));
    
    workSpaceDims[0] = static_cast<int>(workSpaceSize / sizeof(T));
    Tensor<T> workSpace(workSpaceDims);
    
    CHECK_CALL_MIOPEN(miopenFindConvolutionForwardAlgorithm(
            handle.miopenHandle(),
            xDesc, x.data(), wDesc, w.data(),
            convDesc, yDesc, y.data(),
            1, &returnedAlgoCount, &perfResults,
            workSpace.data(), workSpaceSize, false));
    
    workSpaceDims[0] = static_cast<int>(perfResults.memory / sizeof(T));
    workSpace.reset(workSpaceDims);
    
    CHECK_CALL_MIOPEN(miopenConvolutionForward(handle.miopenHandle(),
            &alpha, xDesc, x.data(), 
            wDesc, w.data(), convDesc,
            perfResults.fwd_algo,
            &beta, yDesc, y.data(),
            workSpace.data(), perfResults.memory));

    if (bias != nullptr) {
        miopenTensorDescriptor_t bDesc;

        CHECK_CALL_MIOPEN(miopenCreateTensorDescriptor(&bDesc));
        CHECK_CALL_MIOPEN(miopenSet4dTensorDescriptor(bDesc,
                miopenFloat, 1, bias->dim(0), 1, 1));
        CHECK_CALL_MIOPEN(miopenConvolutionForwardBias(
                handle.miopenHandle(),
                &alpha, bDesc, bias->data(),
                &beta, yDesc, y.data()));
        CHECK_CALL_MIOPEN(miopenDestroyTensorDescriptor(bDesc));
    }

    CHECK_CALL_MIOPEN(miopenDestroyConvolutionDescriptor(convDesc));
    CHECK_CALL_MIOPEN(miopenDestroyTensorDescriptor(xDesc));
    CHECK_CALL_MIOPEN(miopenDestroyTensorDescriptor(wDesc));
    CHECK_CALL_MIOPEN(miopenDestroyTensorDescriptor(yDesc));
    handle.streamSynchronize();
}

template<typename T>
void ConvolutionOp<T>::ConvBackwardWeight(HipHandle& handle,
        ConvDescriptor& convSpec, const Tensor<T>& dy, 
        const Tensor<T>& x, Tensor<T>& dw, Tensor<T>* dbias){
    
    if (convSpec.mode == "conv") {
        if (convSpec.dilation.size() != 0) {
            CHECK_ARGS(convSpec.dilation[0] == 1 &&
                    convSpec.dilation[1] == 1,
                    "Invalid dilation for convolution!");
        }
        convSpec.dilation.push_back(1);
        convSpec.dilation.push_back(1);
    }

    std::vector<int> workSpaceDims = {0};
    const T alpha = 1.0;
    const T beta = 0.0;
    
    miopenTensorDescriptor_t dyDesc, dwDesc, xDesc;
    miopenConvolutionDescriptor_t convDesc;
    int returnedAlgoCount;
    miopenConvAlgoPerf_t perfResults;
    size_t workSpaceSize;
    
    CHECK_CALL_HIP(hipSetDevice(handle.deviceId()));
    CHECK_CALL_MIOPEN(miopenCreateTensorDescriptor(&dyDesc));
    CHECK_CALL_MIOPEN(miopenCreateTensorDescriptor(&dwDesc));
    CHECK_CALL_MIOPEN(miopenCreateTensorDescriptor(&xDesc));
    
    CHECK_CALL_MIOPEN(miopenSet4dTensorDescriptor(dyDesc, miopenFloat,
            dy.dim(0), dy.dim(1), dy.dim(2), dy.dim(3)));
    CHECK_CALL_MIOPEN(miopenSet4dTensorDescriptor(dwDesc, miopenFloat,
            dw.dim(0), dw.dim(1), dw.dim(2), dw.dim(3)));
    CHECK_CALL_MIOPEN(miopenSet4dTensorDescriptor(xDesc, miopenFloat,
            x.dim(0), x.dim(1), x.dim(2), x.dim(3)));
            
    CHECK_CALL_MIOPEN(miopenCreateConvolutionDescriptor(&convDesc));
    CHECK_CALL_MIOPEN(miopenInitConvolutionDescriptor(convDesc,
            convSpec.getMode(),
            convSpec.padding[0], convSpec.padding[1],
            convSpec.stride[0], convSpec.stride[1],
            convSpec.dilation[0], convSpec.dilation[1]));
    
    // Start Backward Weight
    CHECK_CALL_MIOPEN(miopenConvolutionBackwardWeightsGetWorkSpaceSize(
            handle.miopenHandle(),
            dyDesc, xDesc, convDesc, dwDesc,
            &workSpaceSize));
    
    workSpaceDims[0] = static_cast<int>(workSpaceSize / sizeof(T));
    Tensor<T> workSpace(workSpaceDims);
    
    CHECK_CALL_MIOPEN(miopenFindConvolutionBackwardWeightsAlgorithm(
            handle.miopenHandle(),
            dyDesc, dy.data(), xDesc, x.data(),
            convDesc, dwDesc, dw.data(),
            1, &returnedAlgoCount, &perfResults,
            workSpace.data(), workSpaceSize, false));
    
    workSpaceSize = perfResults.memory;
    workSpaceDims[0] = static_cast<int>(workSpaceSize / sizeof(T));
    workSpace.reset(workSpaceDims);
    
    CHECK_CALL_MIOPEN(miopenConvolutionBackwardWeights(handle.miopenHandle(),
            &alpha, dyDesc, dy.data(), 
            xDesc, x.data(), convDesc,
            perfResults.bwd_weights_algo,
            &beta, dwDesc, dw.data(),
            workSpace.data(), workSpaceSize));

    if (dbias != nullptr) {
        miopenTensorDescriptor_t dbDesc;

        CHECK_CALL_MIOPEN(miopenCreateTensorDescriptor(&dbDesc));
        CHECK_CALL_MIOPEN(miopenSet4dTensorDescriptor(dbDesc,
                miopenFloat, 1, dbias->dim(0), 1, 1));
        CHECK_CALL_MIOPEN(miopenConvolutionBackwardBias(
                handle.miopenHandle(),
                &alpha, dyDesc, dy.data(),
                &beta, dbDesc, dbias->data()));
        CHECK_CALL_MIOPEN(miopenDestroyTensorDescriptor(dbDesc));
    }
       
    CHECK_CALL_MIOPEN(miopenDestroyConvolutionDescriptor(convDesc));
    CHECK_CALL_MIOPEN(miopenDestroyTensorDescriptor(dyDesc));
    CHECK_CALL_MIOPEN(miopenDestroyTensorDescriptor(dwDesc));
    CHECK_CALL_MIOPEN(miopenDestroyTensorDescriptor(xDesc));
    handle.streamSynchronize();
}

template<typename T>
void ConvolutionOp<T>::ConvBackwardData(HipHandle& handle,
        ConvDescriptor& convSpec, const Tensor<T>& dy,
        const Tensor<T>& w, Tensor<T>& dx){
    
    if (convSpec.mode == "conv") {
        if (convSpec.dilation.size() != 0) {
            CHECK_ARGS(convSpec.dilation[0] == 1 &&
                    convSpec.dilation[1] == 1,
                    "Invalid dilation for convolution!");
        }
        convSpec.dilation.push_back(1);
        convSpec.dilation.push_back(1);
    }

    std::vector<int> workSpaceDims = {0};
    const T alpha = 1.0;
    const T beta = 0.0;
    
    miopenTensorDescriptor_t dyDesc, dxDesc, wDesc;
    miopenConvolutionDescriptor_t convDesc;
    int returnedAlgoCount;
    miopenConvAlgoPerf_t perfResults;
    size_t workSpaceSize;
    
    CHECK_CALL_HIP(hipSetDevice(handle.deviceId()));
    CHECK_CALL_MIOPEN(miopenCreateTensorDescriptor(&dyDesc));
    CHECK_CALL_MIOPEN(miopenCreateTensorDescriptor(&dxDesc));
    CHECK_CALL_MIOPEN(miopenCreateTensorDescriptor(&wDesc));
    
    CHECK_CALL_MIOPEN(miopenSet4dTensorDescriptor(dyDesc, miopenFloat,
            dy.dim(0), dy.dim(1), dy.dim(2), dy.dim(3)));
    CHECK_CALL_MIOPEN(miopenSet4dTensorDescriptor(dxDesc, miopenFloat,
            dx.dim(0), dx.dim(1), dx.dim(2), dx.dim(3)));
    CHECK_CALL_MIOPEN(miopenSet4dTensorDescriptor(wDesc, miopenFloat,
            w.dim(0), w.dim(1), w.dim(2), w.dim(3)));

    CHECK_CALL_MIOPEN(miopenCreateConvolutionDescriptor(&convDesc));
    CHECK_CALL_MIOPEN(miopenInitConvolutionDescriptor(convDesc,
            convSpec.getMode(),
            convSpec.padding[0], convSpec.padding[1],
            convSpec.stride[0], convSpec.stride[1],
            convSpec.dilation[0], convSpec.dilation[1]));
    
    CHECK_CALL_MIOPEN(miopenConvolutionBackwardDataGetWorkSpaceSize(
            handle.miopenHandle(),
            dyDesc, wDesc, convDesc, dxDesc,
            &workSpaceSize));
    
    workSpaceDims[0] = static_cast<int>(workSpaceSize / sizeof(T));
    Tensor<T> workSpace(workSpaceDims);
    
    CHECK_CALL_MIOPEN(miopenFindConvolutionBackwardDataAlgorithm(
            handle.miopenHandle(),
            dyDesc, dy.data(), wDesc, w.data(),
            convDesc, dxDesc, dx.data(),
            1, &returnedAlgoCount, &perfResults,
            workSpace.data(), workSpaceSize, false));
    
    workSpaceSize = perfResults.memory;
    workSpaceDims[0] = static_cast<int>(workSpaceSize / sizeof(T));
    workSpace.reset(workSpaceDims);
    
    CHECK_CALL_MIOPEN(miopenConvolutionBackwardData(handle.miopenHandle(),
            &alpha, dyDesc, dy.data(), 
            wDesc, w.data(), convDesc,
            perfResults.bwd_data_algo,
            &beta, dxDesc, dx.data(),
            workSpace.data(), workSpaceSize));
    
    CHECK_CALL_MIOPEN(miopenDestroyConvolutionDescriptor(convDesc));
    CHECK_CALL_MIOPEN(miopenDestroyTensorDescriptor(dyDesc));
    CHECK_CALL_MIOPEN(miopenDestroyTensorDescriptor(wDesc));
    CHECK_CALL_MIOPEN(miopenDestroyTensorDescriptor(dxDesc));
    handle.streamSynchronize();
}

template class ConvolutionOp<float>;
