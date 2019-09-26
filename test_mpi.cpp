#include "test_helper.hpp"
#include "test_mpi.hpp"

template<typename T>
void doAllreduce(Tensor<T>* send, Tensor<T>* recv, MPI_Op opType,
        MPI_Comm& mpiWorld){
    int count = recv->size();
    const void* pSend = (send == nullptr)
        ? MPI_IN_PLACE : send->data();
#ifdef USE_COPY
    void* pRecv = malloc(count * sizeof(T));
    CHECK_CALL_HIP(hipMemcpy(pRecv, recv->data(),
            count * sizeof(T), hipMemcpyDeviceToHost));
#else
    void* pRecv = recv->data();
#endif
    T sample;
    MPI_Request req;
    std::vector<MPI_Request> reqList;
    CHECK_CALLMPI(MPI_Iallreduce(pSend, pRecv, count, 
            toMpiDataType(sample), opType, mpiWorld, &req));
    reqList.push_back(req);
    CHECK_CALLMPI(MPI_Waitall(reqList.size(),
            reqList.data(), MPI_STATUSES_IGNORE));
#ifdef USE_COPY
    CHECK_CALL_HIP(hipMemcpy(recv->data(), pRecv,
            count * sizeof(T), hipMemcpyHostToDevice));
    free(pRecv);
#endif
}

int main(int argc, char** argv){
    Communicator comm(argc, argv);
    HipHandle hipHandle(comm.getRank());

    TimeLogger timeLogger;
    uint32_t timeGap = 0;
    size_t testSizeIn = 8192;
    if(argc > 1)
        testSizeIn = atoi(argv[1]);
    const size_t testSize = testSizeIn;
    std::vector<int> shape(1, testSize);
    Tensor<int> send(1, shape);

    Tensor<int>* NullTensor = nullptr;
    timeLogger.record();
    doAllreduce(NullTensor, &send, MPI_SUM, comm.getWorld());
    timeGap += timeLogger.getGapNow();
    send += 80;
    timeLogger.record();
    doAllreduce(NullTensor, &send, MPI_SUM, comm.getWorld());
    timeGap += timeLogger.getGapNow();
    send *= 32;

    std::vector<int> real(testSize, (1 * comm.getWorldSize() + 80) *
            comm.getWorldSize() * 32);
    std::string test_name = "MPI_test_allreduce-";
#ifdef USE_COPY
    test_name += "cpy-";
#endif
    char pid[10] {0};
    sprintf(pid, "%d", getpid());
    test_name += std::string(pid);
    testSame(send, real, test_name);
    std::cout << test_name << " time: "<< timeGap << " us" << std::endl;
    return 0;
}
