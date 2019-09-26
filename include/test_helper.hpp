#ifndef TEST_HELPER_HPP
#define TEST_HELPER_HPP

#ifndef __HIP_PLATFORM_HCC__
#define __HIP_PLATFORM_HCC__
#endif

#include <assert.h>
#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>

#include <hip/hip_runtime.h>
#include <hip/hip_runtime_api.h>

#include <vector>
#include <numeric>
#include <string>
#include <type_traits>
#include <cstdlib>
#include <iostream>
#include <sstream>
#include <cmath>
#include <memory>
#include <functional>

#define CHECK_ARGS(Expr, Message) \
    { \
        if(!(Expr)) { \
            std::cerr << "[" << __FILE__ << ":" << __LINE__ << "] ";\
            std::cerr << "Error: " << #Message << std::endl; \
            exit(1); \
        } \
    }

#define CHECK_CALL_HIP(Expr) \
    { \
        hipError_t ret = Expr; \
        if (ret != hipSuccess) { \
            std::cerr << "[" << __FILE__ << ":" << __LINE__ << "] ";\
            std::cerr << "Hip Error: " << hipGetErrorString(ret) \
                << std::endl; \
            exit(1);\
        } \
    }

#define CHECK_CALL_MIOPEN(Expr) \
    { \
        miopenStatus_t ret = Expr; \
        if (miopenStatusSuccess != ret) { \
            std::cerr << "[" << __FILE__ << ":" << __LINE__ << "] ";\
            std::cerr << "MIOpen Error: " << miopenGetErrorString(ret) \
                << std::endl; \
            exit(1);\
        } \
    }

#define CHECK_CALL_HIPBLAS(Expr) \
    { \
        hipblasStatus_t ret = Expr; \
        if (ret != HIPBLAS_STATUS_SUCCESS) { \
            std::cerr << "[" << __FILE__ << ":" << __LINE__ << "] ";\
            std::cerr << "HipBlas Execution Error!" << std::endl; \
            exit(1);\
        } \
    }

#include "test_handle.hpp"
#include "test_tensor.hpp"
#include "test_time_logger.hpp"

#endif
