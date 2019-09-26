#ifndef TEST_TENSOR_FUNCTIONS_HPP

#ifndef AFTER_DECLARE
#define OPERATOR_FUNC(Func) \
    { \
        CHECK_ARGS(devPtr_.get() != nullptr, \
                "Error: Invalid operation due to nullptr!"); \
        size_t blockSize = 256; \
        size_t gridSize = size_ > blockSize ? (size_ + 255) / 256 : 1; \
        Func<T, D> opClass; \
        hipStream_t stream; \
        hipStreamCreate(&stream); \
        hipLaunchKernelGGL((scalarOpKernel<T, D, Func<T, D>>), \
                dim3(gridSize), dim3(blockSize), 0, stream, \
                opClass, devPtr_.get(), b, size_); \
        hipStreamDestroy(stream); \
        return *this; \
    }

template<typename T, typename D>
struct Add{
    T& operator()(T& src, const D& tg){
        src += static_cast<T>(tg);
        return src;
    }
};

template<typename T, typename D>
struct Sub{
    T& operator()(T& src, const D& tg){
        src -= static_cast<T>(tg);
        return src;
    }
};

template<typename T, typename D>
struct Mul{
    T& operator()(T& src, const D& tg){
        src *= static_cast<T>(tg);
        return src;
    }
};

template<typename T, typename D>
struct Div{
    T& operator()(T& src, const D& tg){
        src /= static_cast<T>(tg);
        return src;
    }
};

template<typename T, typename D, typename Func>
__global__ void scalarOpKernel(Func op, T* src, D tg, size_t borderSize){
    size_t globalId = blockIdx.x * blockDim.x + threadIdx.x;
    if(globalId < borderSize)
        op(src[globalId], tg);
}

#endif

#ifdef AFTER_DECLARE
#define TEST_TENSOR_FUNCTIONS_HPP
template<typename T, typename D>
void testSame(Tensor<T>& a, const std::vector<D>& b, std::string test_name) {
    std::ostringstream msg;
    if (!a.equal(b, msg, true)) {
        std::cerr << test_name << " Test Failed:" << std::endl;
        std::cerr << msg.str() << std::endl;
        // exit(1);
    } else {
        std::cerr << test_name << " Test Passed!" << std::endl;
    }
}

template<typename T, typename D>
void testSame(Tensor<T>& a, Tensor<D>& b, std::string test_name) {
    std::ostringstream msg;
    if (!a.equal(b, msg, true)) {
        std::cerr << test_name << " Test Failed:" << std::endl;
        std::cerr << msg.str() << std::endl;
        // exit(1);
    } else {
        std::cerr << test_name << " Test Passed!" << std::endl;
    }
}
#endif

#endif
