#include "test_helper.hpp"
    
struct DeconvDescriptor {
    int deconvdim = 2;
    int group = 1;
    std::vector<int> padding;
    std::vector<int> stride;
    std::vector<int> dilation;
    DeconvDescriptor(int padding_h, int padding_w,
            int stride_h, int stride_w,
            int dilation_h, int dilation_w){
        padding.push_back(padding_h);
        padding.push_back(padding_w);
        stride.push_back(stride_h);
        stride.push_back(stride_w);
        dilation.push_back(dilation_h);
        dilation.push_back(dilation_w);
    }
};

template<typename T>
void testDeconvBackward(HipHandle& handle,
        DeconvDescriptor& deconvSpec, const T stdError, 
        const std::vector<T>& dy, std::vector<int>& dySpec,
        const std::vector<T>& x, std::vector<int>& xSpec,
        const std::vector<T>& w, std::vector<int>& wSpec,
        const std::vector<T>& dw, std::vector<int>& dwSpec,
        const std::vector<T>& db, std::vector<int>& dbSpec,
        const std::vector<T>& dx, std::vector<int>& dxSpec){
    Tensor<T> dyData(T(1), dySpec);
    Tensor<T> xData(x, xSpec);
    Tensor<T> wData(w, wSpec);
    Tensor<T> dwData(stdError, dwSpec);
    Tensor<T> dxData(dxSpec);
    std::cerr << "Check delta_weight-init_value:" << std::endl;
    std::cerr << dwData << std::endl;
    std::vector<int> workSpaceDims = {0};
    const T alpha = 1.0;
    const T beta = 0.0;
    
    miopenTensorDescriptor_t dyDesc, dxDesc, dwDesc, xDesc, wDesc;
    miopenConvolutionDescriptor_t deconvDesc;
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
            
    CHECK_CALL_MIOPEN(miopenCreateConvolutionDescriptor(&deconvDesc));
    CHECK_CALL_MIOPEN(miopenInitConvolutionDescriptor(deconvDesc,
            miopenTranspose,
            deconvSpec.padding[0], deconvSpec.padding[1],
            deconvSpec.stride[0], deconvSpec.stride[1],
            deconvSpec.dilation[0], deconvSpec.dilation[1]));
    
    // Start Backward Weight
    CHECK_CALL_MIOPEN(miopenConvolutionBackwardWeightsGetWorkSpaceSize(
            handle.miopenHandle(),
            dyDesc, xDesc, deconvDesc, dwDesc,
            &workSpaceSize));
    
    workSpaceDims[0] = static_cast<int>(workSpaceSize / sizeof(T));
    Tensor<T> workSpace(workSpaceDims);
    
    CHECK_CALL_MIOPEN(miopenFindConvolutionBackwardWeightsAlgorithm(
            handle.miopenHandle(),
            dyDesc, dyData.data(), xDesc, xData.data(),
            deconvDesc, dwDesc, dwData.data(),
            1, &returnedAlgoCount, &perfResults,
            workSpace.data(), workSpaceSize, false));
    
    workSpaceSize = perfResults.memory;
    workSpaceDims[0] = static_cast<int>(workSpaceSize / sizeof(T));
    workSpace.reset(workSpaceDims);
    
    CHECK_CALL_MIOPEN(miopenConvolutionBackwardWeights(handle.miopenHandle(),
            &alpha, dyDesc, dyData.data(), 
            xDesc, xData.data(), deconvDesc,
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
        testSame(dbData, db, std::string("Backward delta_bias-dbData"));
        CHECK_CALL_MIOPEN(miopenDestroyTensorDescriptor(dbDesc));
    }
       
    CHECK_CALL_MIOPEN(miopenDestroyTensorDescriptor(dwDesc));
    CHECK_CALL_MIOPEN(miopenDestroyTensorDescriptor(xDesc));
    handle.streamSynchronize();
    // End Backward Weight
    
    testSame(dwData, dw, std::string("Backward delta_weight-dwData"));
}

int main(int argc, char** argv) {

    CHECK_CALL_HIP(hipFree(0));
    
    float stdError = 0.5;
    if(argc > 1) stdError = atof(argv[1]);

    std::vector<float> x_array = {-0.6671,  1.7192, -1.2086,  1.2454,
        -0.6652, -0.9885,  0.0499,  0.1964,
        -0.1973,  1.2310,  0.1152,  0.7395,
        -0.8608,  2.1752, -1.3127,  0.7532,
        -0.4895,  1.8831,  0.6865, -0.9189,
        1.2049, -1.6723, -0.4515, -1.2286,
        -0.9865, -0.9323, -0.4145, -0.5130,
        -0.1131,  0.2655, -2.4192, -1.3803,
        0.3083,  1.2565, -0.9073,  0.2407,
        -1.9617, -0.0832,  1.1119, -0.6040,
        0.2878,  2.2395,  0.3418, -1.4211,
        -0.2965, -0.8339,  1.7215, -0.7722,
        -0.4381, -0.2254,  0.6824,  1.8916,
        0.3967, -2.1019,  0.6397,  1.5591,
        -0.5058,  2.0684, -0.6783,  1.2788,
        1.0566, -2.0563,  0.0228,  0.1780,
        1.6386,  1.6733,  0.8226, -0.5406,
        1.8118, -1.6473, -0.1764,  0.8936,
        -0.9256,  0.4544,  0.5450,  0.4780,
        0.9971, -0.2033,  0.6492, -0.4356,
        0.5565,  1.7077,  1.9988,  0.2595,
        -0.3809, -1.1851,  0.2242, -0.4854,
        1.7408,  0.6216,  0.9197,  0.4270,
        -0.2480,  0.0658,  0.0719, -0.0754};
    std::vector<int> x_array_shape = {2, 3, 4, 4};
    
    std::vector<float> w_array = {-0.1821, 0.5005, 0.4823,
        -0.5078, 0.0187, 0.4072};
    std::vector<int> w_array_shape = {3, 2, 1, 1}; 
    
    std::vector<float> y_std = {-0.1088,  0.6186,  0.5342, -0.6655,
        0.6656, -0.6281, -0.2061, -0.6396,
        -0.4345, -0.6319, -0.2145, -0.4087,
        0.0967, -0.2836, -0.8955, -0.8173,
        0.0402,  0.4159, -1.3230,  1.1880,
        -1.7436,  0.3206,  0.7070,  0.4762,
        0.5194,  2.0015,  0.4073,  0.0519,
        -0.4941,  0.6143,  1.2725,  0.7635,
        0.8805,  0.8800,  0.3099, -0.6003,
        0.7945, -0.4339, -0.1974,  0.1380,
        -0.3218, -0.1459,  0.4036,  0.0057,
        0.2839,  0.2776,  0.3103, -0.2439,
        -0.8247, -0.2671,  0.7377,  1.3269,
        -0.8766, -0.6981,  0.5010,  0.1289,
        0.9257,  1.0576, -0.2417,  0.5712,
        -0.0785, -0.8991, -0.2890,  0.2796};
    std::vector<int> y_std_shape = {2, 2, 4, 4};
    
    std::vector<float> w_grad_std = {6.0931, 6.0931, -1.4449,
        -1.4449, 6.8468, 6.8468};
    std::vector<int> w_grad_std_shape = {3, 2, 1, 1};

    std::vector<float> x_grad_std = {0.3184,  0.3184,  0.3184,  0.3184,
        0.3184,  0.3184,  0.3184,  0.3184,
        0.3184,  0.3184,  0.3184,  0.3184,
        0.3184,  0.3184,  0.3184,  0.3184,
        -0.0255, -0.0255, -0.0255, -0.0255,
        -0.0255, -0.0255, -0.0255, -0.0255,
        -0.0255, -0.0255, -0.0255, -0.0255,
        -0.0255, -0.0255, -0.0255, -0.0255,
        0.4259,  0.4259,  0.4259,  0.4259,
        0.4259,  0.4259,  0.4259,  0.4259,
        0.4259,  0.4259,  0.4259,  0.4259,
        0.4259,  0.4259,  0.4259,  0.4259,
        0.3184,  0.3184,  0.3184,  0.3184,
        0.3184,  0.3184,  0.3184,  0.3184,
        0.3184,  0.3184,  0.3184,  0.3184,
        0.3184,  0.3184,  0.3184,  0.3184,
        -0.0255, -0.0255, -0.0255, -0.0255,
        -0.0255, -0.0255, -0.0255, -0.0255,
        -0.0255, -0.0255, -0.0255, -0.0255,
        -0.0255, -0.0255, -0.0255, -0.0255,
        0.4259,  0.4259,  0.4259,  0.4259,
        0.4259,  0.4259,  0.4259,  0.4259,
        0.4259,  0.4259,  0.4259,  0.4259,
        0.4259,  0.4259,  0.4259,  0.4259};
    std::vector<int> x_grad_std_shape = {2, 3, 4, 4};

    std::vector<float> b_grad_std = {32., 32.};
    std::vector<int> b_grad_std_shape = {2};
    
    DeconvDescriptor deconvDescTrivial(0, 0, 1, 1, 1, 1);
    
    HipHandle handle(0);
    
    testDeconvBackward(handle, deconvDescTrivial,
            stdError, y_std, y_std_shape,
            x_array, x_array_shape,
            w_array, w_array_shape,
            w_grad_std, w_grad_std_shape,
            b_grad_std, b_grad_std_shape,
            x_grad_std, x_grad_std_shape);
    
    return 0;

}
