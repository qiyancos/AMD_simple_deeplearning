#ifndef TEST_OPERATORS_HPP
#define TEST_OPERATORS_HPP

#include "test_helper.hpp"
#include "test_descriptors.hpp"
#include "test_operators_funtions.hpp"

#define HIP_GETTID() ((blockIdx.y * blockDim.y + threadIdx.y) \
        * (gridDim.x * blockDim.x) + (blockIdx.x * blockDim.x + threadIdx.x))

// Macro for Instance Generation
#define GEN_INSTANCE(ClassName) \
    ClassName(uint8_t); \
    ClassName(uint16_t); \
    ClassName(uint32_t); \
    ClassName(float); \
    ClassName(double);

// Convolution Ops
template<typename T>
class ConvolutionOp {
public:
    static void ConvForward(HipHandle& handle, ConvDescriptor& convSpec,
            const Tensor<T>& x, const Tensor<T>& w,
            const Tensor<T>* bias, Tensor<T>& y);

    static void ConvBackwardWeight(HipHandle& handle, ConvDescriptor& convSpec,
            const Tensor<T>& dy, const Tensor<T>& x,
            Tensor<T>& dw, Tensor<T>* dbias);

    static void ConvBackwardData(HipHandle& handle, ConvDescriptor& convSpec,
            const Tensor<T>& dy, const Tensor<T>& w, Tensor<T>& dx);
};

// Deconvolution Ops
template<typename T>
class DeconvolutionOp {
public:
    static void DeconvForward(HipHandle& handle, ConvDescriptor& convSpec,
            const Tensor<T>& x, const Tensor<T>& w,
            const Tensor<T>* bias, Tensor<T>& y);

    static void DeconvBackwardWeight(HipHandle& handle, ConvDescriptor& convSpec,
            const Tensor<T>& dy, const Tensor<T>& x,
            Tensor<T>& dw, Tensor<T>* dbias);

    static void DeconvBackwardData(HipHandle& handle, ConvDescriptor& convSpec,
            const Tensor<T>& dy, const Tensor<T>& w, Tensor<T>& dx);
};

// Pooling Ops
template<typename T>
class PoolingOp {
public:
    static void PoolingForward(HipHandle& handle, PoolingDescriptor& poolSpec,
            const Tensor<T>& x, Tensor<T>& y);
    static void PoolingBackward(HipHandle& handle, PoolingDescriptor& poolSpec,
            const Tensor<T>& x, const Tensor<T>& y,
            const Tensor<T>& dy, Tensor<T>& dx);
};

// FullyConnect Ops
template <typename T>
class FullyConnectOp {
public:
    static void FullyConnectForward(HipHandle& handle,
            const Tensor<T>& x, const Tensor<T>& w,
            const Tensor<T>* bias, Tensor<T>& y);
    static void FullyConnectBackwardWeight(HipHandle& handle,
            const Tensor<T>& dy, const Tensor<T>& x,
            Tensor<T>& dw, Tensor<T>* dbias);
    static void FullyConnectBackwardData(HipHandle& handle,
            const Tensor<T>& dy, const Tensor<T>& w,
            Tensor<T>& dx);
};

#endif
