#include "test_helper.hpp"
#include "test_operators.hpp"

int main(int argc, char** argv){
    HipHandle handle(0);
    CHECK_CALL_HIP(hipSetDevice(0));
    Tensor<float> test_x(1, std::vector<int>(4, 1));
    Tensor<float> test_y(2, std::vector<int>(4, 1));
    Tensor<float> result(std::vector<int>(4, 1));
    OperatorsFunc<float>::dotImpl(handle, 1, test_x, test_y, result);
    testSame(result, std::vector<float>(1, 2), "Dot Result");
    return 0;
}
