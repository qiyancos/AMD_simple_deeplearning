#include "test_helper.hpp"
#include "test_mpi.hpp"

struct ConvDescriptor {
    int convdim = 2;
    int group = 1;
    std::vector<int> padding;
    std::vector<int> stride;
    ConvDescriptor(int padding_h, int padding_w,
            int stride_h, int stride_w){
        padding.push_back(padding_h);
        padding.push_back(padding_w);
        stride.push_back(stride_h);
        stride.push_back(stride_w);
    }
};

template<typename T>
void testConvBackwardDummy(HipHandle& handle,
        ConvDescriptor& convSpec, 
        std::vector<int>& dySpec, std::vector<int>& xSpec,
        std::vector<int>& wSpec, std::vector<int>& dwSpec,
        std::vector<int>& dbSpec, std::vector<int>& dxSpec){
    Tensor<T> dyData(T(1), dySpec);
    Tensor<T> xData(T(1), xSpec);
    Tensor<T> wData(T(1), wSpec);
    Tensor<T> dwData(dwSpec);
    Tensor<T> dxData(dxSpec);
    std::vector<int> workSpaceDims = {0};
    const T alpha = 1.0;
    const T beta = 0.0;
    
    miopenTensorDescriptor_t dyDesc, dxDesc, dwDesc, xDesc, wDesc;
    miopenConvolutionDescriptor_t convDesc;
    int returnedAlgoCount;
    miopenConvAlgoPerf_t perfResults;
    size_t workSpaceSize;
    
    CHECK_CALL_HIP(hipSetDevice(handle.deviceId()));
    CHECK_CALL_MIOPEN(miopenCreateTensorDescriptor(&dyDesc));
    CHECK_CALL_MIOPEN(miopenCreateTensorDescriptor(&dxDesc));
    CHECK_CALL_MIOPEN(miopenCreateTensorDescriptor(&dwDesc));
    CHECK_CALL_MIOPEN(miopenCreateTensorDescriptor(&xDesc));
    CHECK_CALL_MIOPEN(miopenCreateTensorDescriptor(&wDesc));
    
    CHECK_CALL_MIOPEN(miopenSet4dTensorDescriptor(dyDesc, miopenFloat,
            dySpec[0], dySpec[1], dySpec[2], dySpec[3]));
    CHECK_CALL_MIOPEN(miopenSet4dTensorDescriptor(dxDesc, miopenFloat,
            dxSpec[0], dxSpec[1], dxSpec[2], dxSpec[3]));
    CHECK_CALL_MIOPEN(miopenSet4dTensorDescriptor(dwDesc, miopenFloat,
            dwSpec[0], dwSpec[1], dwSpec[2], dwSpec[3]));
    CHECK_CALL_MIOPEN(miopenSet4dTensorDescriptor(wDesc, miopenFloat,
            wSpec[0], wSpec[1], wSpec[2], wSpec[3]));
    CHECK_CALL_MIOPEN(miopenSet4dTensorDescriptor(xDesc, miopenFloat,
            xSpec[0], xSpec[1], xSpec[2], xSpec[3]));
            
    CHECK_CALL_MIOPEN(miopenCreateConvolutionDescriptor(&convDesc));
    CHECK_CALL_MIOPEN(miopenInitConvolutionDescriptor(convDesc,
            miopenConvolution,
            convSpec.padding[0], convSpec.padding[1],
            convSpec.stride[0], convSpec.stride[1],
            1, 1));
    
    // Start Backward Weight
    CHECK_CALL_MIOPEN(miopenConvolutionBackwardWeightsGetWorkSpaceSize(
            handle.miopenHandle(),
            dyDesc, xDesc, convDesc, dwDesc,
            &workSpaceSize));
    
    workSpaceDims[0] = static_cast<int>(workSpaceSize / sizeof(T));
    Tensor<T> workSpace(workSpaceDims);
    
    CHECK_CALL_MIOPEN(miopenFindConvolutionBackwardWeightsAlgorithm(
            handle.miopenHandle(),
            dyDesc, dyData.data(), xDesc, xData.data(),
            convDesc, dwDesc, dwData.data(),
            1, &returnedAlgoCount, &perfResults,
            workSpace.data(), workSpaceSize, false));
    
    workSpaceSize = perfResults.memory;
    workSpaceDims[0] = static_cast<int>(workSpaceSize / sizeof(T));
    workSpace.reset(workSpaceDims);
    
    CHECK_CALL_MIOPEN(miopenConvolutionBackwardWeights(handle.miopenHandle(),
            &alpha, dyDesc, dyData.data(), 
            xDesc, xData.data(), convDesc,
            perfResults.bwd_weights_algo,
            &beta, dwDesc, dwData.data(),
            workSpace.data(), workSpaceSize));

    if (-1 != dbSpec[0]) {
        Tensor<T> dbData(dbSpec);
        miopenTensorDescriptor_t dbDesc;

        CHECK_CALL_MIOPEN(miopenCreateTensorDescriptor(&dbDesc));
        CHECK_CALL_MIOPEN(miopenSet4dTensorDescriptor(dbDesc,
                miopenFloat, 1, dbSpec[0], 1, 1));
        CHECK_CALL_MIOPEN(miopenConvolutionBackwardBias(
                handle.miopenHandle(),
                &alpha, dyDesc, dyData.data(),
                &beta, dbDesc, dbData.data()));
        CHECK_CALL_MIOPEN(miopenDestroyTensorDescriptor(dbDesc));
    }
       
    CHECK_CALL_MIOPEN(miopenDestroyTensorDescriptor(dwDesc));
    CHECK_CALL_MIOPEN(miopenDestroyTensorDescriptor(xDesc));
    handle.streamSynchronize();
    // End Backward Weight
    
    // Start Backward Data
    CHECK_CALL_MIOPEN(miopenConvolutionBackwardDataGetWorkSpaceSize(
            handle.miopenHandle(),
            dyDesc, wDesc, convDesc, dxDesc,
            &workSpaceSize));
    
    workSpaceDims[0] = static_cast<int>(workSpaceSize / sizeof(T));
    workSpace.reset(workSpaceDims);
    
    CHECK_CALL_MIOPEN(miopenFindConvolutionBackwardDataAlgorithm(
            handle.miopenHandle(),
            dyDesc, dyData.data(), wDesc, wData.data(),
            convDesc, dxDesc, dxData.data(),
            1, &returnedAlgoCount, &perfResults,
            workSpace.data(), workSpaceSize, false));
    
    workSpaceSize = perfResults.memory;
    workSpaceDims[0] = static_cast<int>(workSpaceSize / sizeof(T));
    workSpace.reset(workSpaceDims);
    
    CHECK_CALL_MIOPEN(miopenConvolutionBackwardData(handle.miopenHandle(),
            &alpha, dyDesc, dyData.data(), 
            wDesc, wData.data(), convDesc,
            perfResults.bwd_data_algo,
            &beta, dxDesc, dxData.data(),
            workSpace.data(), workSpaceSize));
    
    CHECK_CALL_MIOPEN(miopenDestroyConvolutionDescriptor(convDesc));
    CHECK_CALL_MIOPEN(miopenDestroyTensorDescriptor(dyDesc));
    CHECK_CALL_MIOPEN(miopenDestroyTensorDescriptor(wDesc));
    CHECK_CALL_MIOPEN(miopenDestroyTensorDescriptor(dxDesc));
    handle.streamSynchronize();
    // End Backward Data
}

int main(int argc, char** argv){
    Communicator comm(argc, argv);
    HipHandle hipHandle(comm.getRank());

    std::vector<int> y_std_shape = {32, 64, 224, 224};
    std::vector<int> x_array_shape = {32, 3, 224, 224};
    std::vector<int> w_array_shape = {64, 3, 3, 3}; 
    std::vector<int> w_grad_std_shape = {64, 3, 3, 3};
    std::vector<int> x_grad_std_shape = {32, 3, 224, 224};
    std::vector<int> b_grad_std_shape = {-1};
    ConvDescriptor convDescTrivial(1, 1, 3, 3);
    
    int testIters = 20000;
    for(int i = 0; i < testIters; i++){
        std::cout << "Pid-" << getpid() << ": Running Iter " << i << std::endl;
        testConvBackwardDummy<float>(hipHandle, convDescTrivial,
                y_std_shape, x_array_shape, w_array_shape,
                w_grad_std_shape, b_grad_std_shape,
                x_grad_std_shape);
    }

    return 0;
}
