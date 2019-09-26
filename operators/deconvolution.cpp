#include "test_operators.hpp"

// Deconvolution Ops
template<typename T>
void DeconvolutionOp<T>::DeconvForward(HipHandle& handle,
        ConvDescriptor& convSpec,
        const Tensor<T>& x, const Tensor<T>& w,
        const Tensor<T>* bias, Tensor<T>& y){
    CHECK_ARGS(convSpec.mode == "deconv",
            "Invalid mode for deconvolution!");
    CHECK_ARGS(convSpec.dilation.size() == 2,
            "Dilations must be specified for deconvolution!");
    ConvolutionOp<T>::ConvForward(handle, convSpec, x, w, bias, y);
}

template<typename T>
void DeconvolutionOp<T>::DeconvBackwardWeight(HipHandle& handle,
        ConvDescriptor& convSpec,
        const Tensor<T>& dy, const Tensor<T>& x,
        Tensor<T>& dw, Tensor<T>* dbias){
    CHECK_ARGS(convSpec.mode == "deconv",
            "Invalid mode for deconvolution!");
    CHECK_ARGS(convSpec.dilation.size() == 2,
            "Dilations must be specified for deconvolution!");
    ConvolutionOp<T>::ConvBackwardWeight(handle, convSpec, dy, x, dw, dbias);
}

template<typename T>
void DeconvolutionOp<T>::DeconvBackwardData(HipHandle& handle,
        ConvDescriptor& convSpec, const Tensor<T>& dy,
        const Tensor<T>& w, Tensor<T>& dx){
    CHECK_ARGS(convSpec.mode == "deconv",
            "Invalid mode for deconvolution!");
    CHECK_ARGS(convSpec.dilation.size() == 2,
            "Dilations must be specified for deconvolution!");
    ConvolutionOp<T>::ConvBackwardData(handle, convSpec, dy, w, dx);
}

template class DeconvolutionOp<float>;
