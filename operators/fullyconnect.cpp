#include "test_operators.hpp"

template<typename T>
__global__ void hipFullyConnectForwardBiasKernel(
        uint32_t m, uint32_t n, const T *bias, T *y) {
    size_t i = HIP_GETTID();
    if (i >= n) return;

    for (uint32_t j = 0; j < m; ++j) {
        y[j * n + i] += bias[i];
    }
}

template<typename T>
__global__ void hipFullyConnectBackwardBiasKernel(
        uint32_t m, uint32_t n, const T *dy, T *dbias) {
    size_t i = HIP_GETTID();
    if (i >= n) return;

    dbias[i] = T(0);
    for (uint32_t j = 0; j < m; ++j) {
        dbias[i] += dy[j * n + i];
    }
}

// FullyConnect Ops
template <typename T>
void FullyConnectOp<T>::FullyConnectForward(HipHandle& handle,
        const Tensor<T>& x, const Tensor<T>& w,
        const Tensor<T>* bias, Tensor<T>& y){
    const T alpha = 1.0;
    const T beta = 0.0;

    int M = x.dim(0);
    int K = x.size() / x.dim(0);
    int N = w.dim(0);

    OperatorsFunc<T>::gemmImpl(handle, BLAS_OP_T, BLAS_OP_N,
            N, M, K, alpha, w, x, beta, y);

    if (bias) {
        size_t blockSize = 256;
        size_t gridSize = (N + 255) / 256;
        hipLaunchKernelGGL((hipFullyConnectForwardBiasKernel<T>),
                dim3(gridSize), dim3(blockSize), 0, handle.stream(),
                uint32_t(M), uint32_t(N), bias->data(), y.data());
    }
    handle.streamSynchronize();
}

template <typename T>
void FullyConnectOp<T>::FullyConnectBackwardWeight(HipHandle& handle,
        const Tensor<T>& dy, const Tensor<T>& x,
        Tensor<T>& dw, Tensor<T>* dbias){
    const T alpha = 1.0;
    const T beta = 0.0;

    int M = x.dim(0);
    int K = x.size() / x.dim(0);
    int N = dw.dim(0);

    OperatorsFunc<T>::gemmImpl(handle, BLAS_OP_N, BLAS_OP_T,
            K, N, M, alpha, x, dy, beta, dw);

    if (dbias) {
        uint32_t m = dy.dim(0);
        uint32_t n = dy.dim(1);
        size_t blockSize = 256;
        size_t gridSize = (n + 255) / 256;

        hipLaunchKernelGGL((hipFullyConnectBackwardBiasKernel<T>),
                dim3(gridSize), dim3(blockSize), 0, handle.stream(),
                m, n, dy.data(), dbias->data());
    }
    handle.streamSynchronize();
}

template <typename T>
void FullyConnectOp<T>::FullyConnectBackwardData(HipHandle& handle,
        const Tensor<T>& dy, const Tensor<T>& w, Tensor<T>& dx){
    const T alpha = 1.0;
    const T beta = 0.0;

    int M = dx.dim(0);
    int K = dx.size() / dx.dim(0);
    int N = w.dim(0);

    OperatorsFunc<T>::gemmImpl(handle, BLAS_OP_N, BLAS_OP_N,
            K, M, N, alpha, w, dy, beta, dx);
    handle.streamSynchronize();
}

template class FullyConnectOp<float>;
