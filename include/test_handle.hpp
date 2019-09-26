#ifndef TEST_HANDLE_HPP
#define TEST_HANDLE_HPP

#include <hipblas.h>
#include <miopen/miopen.h>

class HipHandle final{
private:
    miopenHandle_t miopenHandle_;
    hipblasHandle_t hipblasHandle_;
    hipStream_t stream_;
    int deviceId_;

public:
    using key_t = size_t;
    using resource_t = HipHandle;

    HipHandle(const HipHandle&) = delete;
    HipHandle(HipHandle&&) = delete;
    HipHandle& operator=(const HipHandle&) = delete;
    HipHandle& operator=(HipHandle&&) = delete;

    HipHandle() {
        new (this) HipHandle(0);
    }

    explicit HipHandle(int deviceId) : deviceId_(deviceId) {
        CHECK_CALL_HIP(hipSetDevice(deviceId));
        CHECK_CALL_HIP(hipStreamCreate(&stream_));
        CHECK_CALL_MIOPEN(miopenCreateWithStream(&miopenHandle_, stream_));
        CHECK_CALL_HIPBLAS(hipblasCreate(&hipblasHandle_));
        CHECK_CALL_HIPBLAS(hipblasSetStream(hipblasHandle_, stream_));
    }

    ~HipHandle() {
        if (miopenHandle_)
            CHECK_CALL_MIOPEN(miopenDestroy(miopenHandle_));
        if (hipblasHandle_)
            CHECK_CALL_HIPBLAS(hipblasDestroy(hipblasHandle_));
        if (stream_)
            CHECK_CALL_HIP(hipStreamDestroy(stream_));
    }

    int deviceId() {return deviceId_;}
    hipStream_t stream() {return stream_;}
    miopenHandle_t miopenHandle() {return miopenHandle_;}
    hipblasHandle_t hipblasHandle() {return hipblasHandle_;}
    void streamSynchronize() { CHECK_CALL_HIP(hipStreamSynchronize(stream_)); }
};

#endif
