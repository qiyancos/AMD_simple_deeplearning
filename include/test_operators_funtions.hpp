#ifndef TEST_OPERATORS_FUNTIONS_HPP
#define TEST_OPERATORS_FUNTIONS_HPP

constexpr char BLAS_OP_T = 'T';
constexpr char BLAS_OP_N = 'N';
constexpr char BLAS_OP_CONJTRANS = 'C';

template<typename T>
class OperatorsFunc {
public:
    static void dotImpl(HipHandle& handle, size_t n,
            const Tensor<T>& x, const Tensor<T>& y,
            Tensor<T>& result);
    
    static void gemvImpl(HipHandle& handle,
            char transa, size_t m, size_t n, T alpha,
            const Tensor<T>& A, const Tensor<T>& x,
            T beta, Tensor<T>& y);
    
    static void gerImpl(HipHandle& handle,
            size_t m, size_t n, T alpha,
            const Tensor<T>& x, const Tensor<T>& y,
            Tensor<T>& A);

    static void gemmImpl(HipHandle& handle,
            char transa, char transb, size_t m, size_t n, size_t k,
            T alpha, const Tensor<T>& A, const Tensor<T>& B,
            T beta, Tensor<T>& C);
    
    static void bgemmImpl(HipHandle& handle,
            char transa, char transb, size_t m, size_t n, size_t k,
            T alpha, const Tensor<T>& A, const Tensor<T>& B,
            T beta, Tensor<T>& C, size_t nbatch);
};

#endif
