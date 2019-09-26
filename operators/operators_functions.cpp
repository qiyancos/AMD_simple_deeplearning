#include "test_operators.hpp"

template<typename T>
void OperatorsFunc<T>::dotImpl(HipHandle& handle, size_t n,
        const Tensor<T>& x, const Tensor<T>& y,
        Tensor<T>& result) {
    CHECK_CALL_HIPBLAS(hipblasSetPointerMode(
            handle.hipblasHandle(), HIPBLAS_POINTER_MODE_DEVICE));
    CHECK_CALL_HIPBLAS(hipblasSdot(
            handle.hipblasHandle(), n, x.data(), 1,
            y.data(), 1, result.data()));
    CHECK_CALL_HIPBLAS(hipblasSetPointerMode(
            handle.hipblasHandle(), HIPBLAS_POINTER_MODE_HOST));
}

template<typename T>
void OperatorsFunc<T>::gemvImpl(HipHandle& handle,
        char transa, size_t m, size_t n, T alpha,
        const Tensor<T>& A, const Tensor<T>& x,
        T beta, Tensor<T>& y) {
    CHECK_ARGS(false, "Unsupported operation!");
}

template<typename T>
void OperatorsFunc<T>::gerImpl(HipHandle& handle,
        size_t m, size_t n, T alpha,
        const Tensor<T>& x, const Tensor<T>& y,
        Tensor<T>& A) {
    CHECK_ARGS(false, "Unsupported operation!");
}

template<typename T> 
void OperatorsFunc<T>::gemmImpl(HipHandle& handle,
        char transa, char transb, size_t m, size_t n, size_t k,
        T alpha, const Tensor<T>& A, const Tensor<T>& B,
        T beta, Tensor<T>& C) {
    CHECK_ARGS(false, "Unsupported operation!");
}

template<typename T>
void OperatorsFunc<T>::bgemmImpl(HipHandle& handle, \
        char transa, char transb, size_t m, size_t n, size_t k,
        T alpha, const Tensor<T>& A, const Tensor<T>& B,
        T beta, Tensor<T>& C, size_t nbatch) {
    CHECK_ARGS(false, "Unsupported operation!");
}

template<>
void OperatorsFunc<float>::dotImpl(HipHandle& handle, size_t n,
        const Tensor<float>& x, const Tensor<float>& y,
        Tensor<float>& result) {
    CHECK_CALL_HIPBLAS(hipblasSetPointerMode(
        handle.hipblasHandle(),
        HIPBLAS_POINTER_MODE_DEVICE));
    // Use hipblasDdot for double
    CHECK_CALL_HIPBLAS(hipblasSdot(
        handle.hipblasHandle(),
        n, x.data(), 1, y.data(), 1, result.data()));
    CHECK_CALL_HIPBLAS(hipblasSetPointerMode(
        handle.hipblasHandle(),
        HIPBLAS_POINTER_MODE_HOST));
}

template<>
void OperatorsFunc<float>::gemvImpl(HipHandle& handle,
        char transa, size_t m, size_t n, float alpha,
        const Tensor<float>& A, const Tensor<float>& x,
        float beta, Tensor<float>& y) {
    CHECK_ARGS(transa == BLAS_OP_T || transa == BLAS_OP_N, 
            "HIPBLAS: Unsupported BLAS_OP");
    hipblasOperation_t hiptransa =
        transa == BLAS_OP_T? HIPBLAS_OP_T : HIPBLAS_OP_N;
    // Use hipblasDdot for double
    CHECK_CALL_HIPBLAS(hipblasSgemv(
        handle.hipblasHandle(),
        hiptransa, m, n, &alpha, A.data(), m,
        x.data(), 1, &beta, y.data(), 1));
}

template<>
void OperatorsFunc<float>::gerImpl(HipHandle& handle,
        size_t m, size_t n, float alpha,
        const Tensor<float>& x, const Tensor<float>& y,
        Tensor<float>& A) {
    // Use hipblasDdot for double
    CHECK_CALL_HIPBLAS(hipblasSger(
        handle.hipblasHandle(), m, n, &alpha, x.data(),
        1, y.data(), 1, A.data(), m));
}

template<> 
void OperatorsFunc<float>::gemmImpl(HipHandle& handle,
        char transa, char transb, size_t m, size_t n, size_t k,
        float alpha, const Tensor<float>& A, const Tensor<float>& B,
        float beta, Tensor<float>& C) {
    CHECK_ARGS((transa == BLAS_OP_T || transa == BLAS_OP_N) &&
            (transb == BLAS_OP_T || transb == BLAS_OP_N),
            "HIPBLAS: Unsupported BLAS_OP");
    int lda = (transa == BLAS_OP_T) ?
              static_cast<int>(k) : static_cast<int>(m);
    int ldb = (transb == BLAS_OP_T) ?
              static_cast<int>(n) : static_cast<int>(k);
    hipblasOperation_t hiptransa =
        transa == BLAS_OP_T? HIPBLAS_OP_T : HIPBLAS_OP_N;
    hipblasOperation_t hiptransb =
        transb == BLAS_OP_T? HIPBLAS_OP_T : HIPBLAS_OP_N;
    // Use hipblasDdot for double
    CHECK_CALL_HIPBLAS(hipblasSgemm(
        handle.hipblasHandle(), hiptransa, hiptransb,
        static_cast<int>(m), static_cast<int>(n), static_cast<int>(k),
        &alpha, A.data(), lda, B.data(), ldb, &beta,
        C.data(), static_cast<int>(m)));
}

template<>
void OperatorsFunc<float>::bgemmImpl(HipHandle& handle, \
        char transa, char transb, size_t m, size_t n, size_t k,
        float alpha, const Tensor<float>& A, const Tensor<float>& B,
        float beta, Tensor<float>& C, size_t nbatch) {
    std::vector<const float *> hbuf(3 * nbatch);
    const float **hptra = hbuf.data();
    const float **hptrb = hptra + nbatch;
    const float **hptrc = hptrb + nbatch;
    for (size_t i = 0; i < nbatch; i ++) {
        hptra[i] = A.data() + i * m * k;
        hptrb[i] = B.data() + i * k * n;
        hptrc[i] = C.data() + i * m * n;
    }
    
    std::vector<int> shape(1, 3 * nbatch);
    Tensor<float> tmp(int(0), shape);
    float **dptra = reinterpret_cast<float **>(tmp.data());
    float **dptrb = dptra + nbatch;
    float **dptrc = dptrb + nbatch;
    
    hipStream_t hipstream = handle.stream();
    CHECK_CALL_HIP(hipMemcpyAsync(dptra, hptra,
        3 * nbatch * sizeof(float *), hipMemcpyHostToDevice, hipstream));
    
    CHECK_ARGS((transa == BLAS_OP_T || transa == BLAS_OP_N) &&
            (transb == BLAS_OP_T || transb == BLAS_OP_N),
            "HIPBLAS: Unsupported BLAS_OP");
    hipblasOperation_t hiptransa =
        transa == BLAS_OP_T? HIPBLAS_OP_T : HIPBLAS_OP_N;
    hipblasOperation_t hiptransb =
        transb == BLAS_OP_T? HIPBLAS_OP_T : HIPBLAS_OP_N;
    size_t lda = hiptransa == HIPBLAS_OP_T ? k : m;
    size_t ldb = hiptransb == HIPBLAS_OP_T ? n : k;
    size_t ldc = m;
    
    // Use hipblasDdot for double
    CHECK_CALL_HIPBLAS(hipblasSgemmBatched(
        handle.hipblasHandle(), hiptransa, hiptransb,
        m, n, k, &alpha, (const float **)dptra, lda, (const float **)dptrb, ldb,
        &beta, dptrc, ldc, nbatch));
}

template class OperatorsFunc<float>;
