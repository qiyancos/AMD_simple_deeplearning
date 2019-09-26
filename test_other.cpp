#include "test_helper.hpp"

void testMemInfo(){
    hipDeviceProp_t prop;
    size_t free, total;
    CHECK_CALL_HIP(hipMemGetInfo(&free, &total));
    CHECK_CALL_HIP(hipGetDeviceProperties(&prop, 0));
    printf("MemInfo(%lu/%lu), MemLimit-%lu\n", total - free,
            total, prop.totalGlobalMem);
}

void testAddress(){
    int times = 4;
    while(times--){
        std::vector<int> shape {1024};
        Tensor<int> tempGPU(1, shape);
        int* tempCPU = static_cast<int*>(malloc(1024 * sizeof(int)));
        printf("Pointer: GPU-%p, CPU-%p\n", tempGPU.data(), tempCPU);
        free(tempCPU);
    }
}

void testOutOfBound(){
    std::vector<int> shape {16};
    Tensor<int> temp(1, shape);
    for(int i = 0; i < 32; i++)
        printf("Test[%d]-%p: %d\n", i, &(temp.data()[i]), temp.data()[i]);
}

int main(){
    CHECK_CALL_HIP(hipSetDevice(0));
    testMemInfo();
    testAddress();
    //testOutOfBound();
    return 0;
}
