#ifndef TEST_TENSOR_HPP
#define TEST_TENSOR_HPP

#include "test_tensor_functions.hpp"
#define FLOATERR 1e-2

template <typename T>
class Tensor final{
private:
    std::shared_ptr<T> devPtr_;
    std::vector<int> dims_;
    hipStream_t hipStream;
    int size_;
    struct deleteDevPtr {
        void operator()(T* p) const {
            hipFree(p);
        }
    };

public:
    Tensor(const Tensor&) = delete;
    Tensor(Tensor&&) = delete;
    Tensor& operator = (const Tensor&) = delete;
    Tensor& operator = (Tensor&&) = delete;

    Tensor() { devPtr_.reset(nullptr); }

    explicit Tensor(const std::vector<int>& dims) : dims_(dims) {
        T* tmpPtr;
        CHECK_ARGS(dims.size() > 0, 
                "Trying to init Tensor with an empty shape!");
        size_ = std::accumulate(dims_.begin(), dims_.end(),
            1, std::multiplies<int>());
        CHECK_ARGS(size_ >= 0, "Trying to init Tensor with an invalid shape!");
        if (size_ == 0) return;
        CHECK_CALL_HIP(hipMalloc(&tmpPtr, sizeof(T) * size_));
        CHECK_CALL_HIP(hipMemset(tmpPtr, 0, sizeof(T) * size_));
        devPtr_.reset(tmpPtr, deleteDevPtr());
    }

    Tensor(T init, const std::vector<int>& dims) : Tensor(dims) {
        this->reset(init);
    }

    Tensor(T* src, const std::vector<int>& dims) : dims_(dims) {
        CHECK_ARGS(src != nullptr, "Trying to init Tensor with a nullptr!");
        T* tmpPtr;
        CHECK_ARGS(dims.size() > 0, 
                "Trying to init Tensor with an empty shape!");
        size_ = std::accumulate(dims_.begin(), dims_.end(),
            1, std::multiplies<int>());
        CHECK_ARGS(size_ >= 0, "Trying to init Tensor with an invalid shape!");
        if (size_ == 0) return;
        CHECK_CALL_HIP(hipMalloc(&tmpPtr, sizeof(T) * size_));
        CHECK_CALL_HIP(hipMemcpy(tmpPtr, src,
                size_ * sizeof(T), hipMemcpyHostToDevice));
        devPtr_.reset(tmpPtr, deleteDevPtr());
    }

    Tensor(const std::vector<T>& src, const std::vector<int>& dims) :
            dims_(dims) {
        T* tmpPtr;
        CHECK_ARGS(dims.size() > 0, 
                "Trying to init Tensor with an empty shape!");
        size_ = std::accumulate(dims_.begin(), dims_.end(),
            1, std::multiplies<int>());
        CHECK_ARGS(size_ == src.size(), 
                "Trying to init Tensor with an invalid shape!");
        CHECK_CALL_HIP(hipMalloc(&tmpPtr, sizeof(T) * size_));
        T* hostPtr = static_cast<T*>(malloc(size_ * sizeof(T)));
        int i = 0;
        for (auto srcIt : src)
            hostPtr[i++] = srcIt;
        CHECK_CALL_HIP(hipMemcpy(tmpPtr, hostPtr,
                size_ * sizeof(T), hipMemcpyHostToDevice));
        free(hostPtr);
        devPtr_.reset(tmpPtr, deleteDevPtr());
    }

    int size() const { return size_; }
    T* data() const { return devPtr_.get();}
    std::vector<int>& dims() const { return dims_; }
    int dim(int nth) const {
        CHECK_ARGS(nth < dims_.size(), "Dim out of range!");
        return dims_[nth];
    }

    void reset() {
        T* tmpPtr;
        CHECK_CALL_HIP(hipMalloc(&tmpPtr, sizeof(T) * size_));
        CHECK_CALL_HIP(hipMemset(tmpPtr, 0, sizeof(T) * size_));
        devPtr_.reset(tmpPtr, deleteDevPtr());
    }

    void reset(const T init) {
        CHECK_ARGS(nullptr != devPtr_.get(), "Cannot reset for nullptr!");
        T* hostPtr = static_cast<T*>(malloc(sizeof(T) * size_));
        for (int i = 0; i < size_; i++)
            hostPtr[i] = init;
        CHECK_CALL_HIP(hipMemcpy(devPtr_.get(), hostPtr,
                sizeof(T) * size_, hipMemcpyHostToDevice));
        free(hostPtr);
    }

    void reset(const std::vector<int>& dims) {
        T* tmpPtr;
        size_ = std::accumulate(dims_.begin(), dims_.end(),
            1, std::multiplies<int>());
        CHECK_CALL_HIP(hipMalloc(&tmpPtr, sizeof(T) * size_));
        CHECK_CALL_HIP(hipMemset(tmpPtr, 0, sizeof(T) * size_));
        devPtr_.reset(tmpPtr, deleteDevPtr());
    }

    template<typename D>
    bool equal(Tensor<D>& b, std::ostringstream& msg, bool info){
        if (!std::is_same<T, D>::value) {
            if (info) msg << "Tensors have different data type!";
            return false;
        }
        if (nullptr == devPtr_.get() || nullptr == b.data()) {
            if (info) msg << "One or two tensors have invalid devPtr!";
            return false;
        }
        for (int i = 0; i < dims_.size(); i++) {
            if (dims_[i] != b.dim(i)) {
                if (info) msg << "Tensors have difference size in dim" << i;
                return false;
            }
        }
        T* hostPtrThis = malloc(size_ * sizeof(T));
        T* hostPtrB = malloc(size_ * sizeof(T));
        CHECK_CALL_HIP(hipMemcpy(hostPtrThis, devPtr_.get(),
                size_ * sizeof(T), hipMemcpyDeviceToHost));
        CHECK_CALL_HIP(hipMemcpy(hostPtrB, b.data(),
                size_ * sizeof(T), hipMemcpyDeviceToHost));
        bool flag = true;
        if (info) msg << "Tensor data does not match: " << std::endl;
        for (int i = 0; i < size_; i++) {
            if (std::abs(hostPtrThis[i] - hostPtrB[i]) > FLOATERR) {
                if (info)
                    flag = false;
                else
                    return false;
            }
            if (info) {
                msg << "id-" << i << ": Tensor A, " << hostPtrThis[i];
                msg << " | Tensor B, " << hostPtrB[i] << std::endl;
            }
        }
        free(hostPtrThis);
        free(hostPtrB);
        return flag;
    }

    template<typename D>
    bool equal(const std::vector<D>& b, std::ostringstream& msg, bool info){
        if (!std::is_same<T, D>::value) {
            if (info) msg << "Tensors have different data type!";
            return false;
        }
        if (nullptr == devPtr_.get()) {
            if (info) msg << "Tensor has invalid devPtr!";
            return false;
        }
        if (size_ != b.size()) {
            if (info) {
                msg << "Tensors have difference size: " << size_;
                msg << " vs. " << b.size();
            }
            return false;
        }
        T* hostPtrThis = static_cast<T*>(malloc(size_ * sizeof(T)));
        CHECK_CALL_HIP(hipMemcpy(hostPtrThis, devPtr_.get(),
                size_ * sizeof(T), hipMemcpyDeviceToHost));
        bool flag = true;
        if (info) msg << "Data does not match: " << std::endl;
        for (int i = 0; i < size_; i++) {
            if (std::abs(hostPtrThis[i] - b[i]) > FLOATERR) {
                if (info)
                    flag = false;
                else
                    return false;
            }
            if (info) {
                msg << "id-" << i << ": Tensor A, " << hostPtrThis[i];
                msg << " | Vetcor B, " << b[i] << std::endl;
            }
        }
        free(hostPtrThis);
        return flag;
    }

    template<typename D>
    bool operator== (Tensor<D>& b) {
        return this->equal(b, nullptr, false);
    }

    template<typename D>
    bool operator== (std::vector<D>& b) {
        return this->equal(b, nullptr, false);
    }

    template<typename D>
    bool operator!= (std::vector<D>& b) {
        return !(*this == b);
    }

    template<typename D>
    bool operator!= (Tensor<D>& b) {
        return !(*this == b);
    }

    template<typename D>
    Tensor<T>& operator+= (D b){
        OPERATOR_FUNC(Add);
    }

    template<typename D>
    Tensor<T>& operator-= (D b){
        OPERATOR_FUNC(Sub);
    }

    template<typename D>
    Tensor<T>& operator*= (D b){
        OPERATOR_FUNC(Mul);
    }

    template<typename D>
    Tensor<T>& operator/= (D b){
        OPERATOR_FUNC(Div);
    }

    template<typename D>
    friend bool operator== (std::vector<D>& a, Tensor<T>& b) { return b == a; }

    template<typename D>
    friend bool operator!= (std::vector<D>& a, Tensor<T>& b) { return b != a; }

    friend std::ostream& operator << (std::ostream& os, Tensor<T>& b) {
        T* hostPtrB = static_cast<T*>(malloc(b.size() * sizeof(T)));
        CHECK_CALL_HIP(hipMemcpy(hostPtrB, b.data(),
                b.size() * sizeof(T), hipMemcpyDeviceToHost));
        for (int i = 0; i < b.size(); i++)
            os << "Tensor-" << i << ": " << hostPtrB[i] << std::endl;
        free(hostPtrB);
        return os;
    }
};

#define AFTER_DECLARE

#include "test_tensor_functions.hpp"

#endif
