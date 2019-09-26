#ifndef TEST_MPI_HPP
#define TEST_MPI_HPP

#include "test_helper.hpp"
#include <mpi.h>

#define CHECK_CALLMPI(Expr) { \
    int ret = Expr; \
    if (ret != MPI_SUCCESS) { \
        std::cerr << "Error: MPI function return failed retcode!" \
                << std::endl; \
        exit(1); \
    } \
}

template<typename T>
inline MPI_Datatype toMpiDataType(T t) {
    if(std::is_same<T, char>::value)
        return MPI_INT8_T;
    if(std::is_same<T, short>::value)
        return MPI_INT16_T;
    if(std::is_same<T, int>::value)
        return MPI_INT32_T;
    if(std::is_same<T, long long>::value)
        return MPI_INT64_T;
    if(std::is_same<T, unsigned char>::value)
        return MPI_UINT8_T;
    if(std::is_same<T, unsigned short>::value)
        return MPI_UINT16_T;
    if(std::is_same<T, unsigned int>::value)
        return MPI_UINT32_T;
    if(std::is_same<T, unsigned long long>::value)
        return MPI_UINT64_T;
    if(std::is_same<T, bool>::value)
        return MPI_C_BOOL;
    if(std::is_same<T, float>::value)
        return MPI_FLOAT;
    if(std::is_same<T, double>::value)
        return MPI_DOUBLE;
    std::cerr << "Error: Unsupported value type!" << std::endl;
    exit(1);
}

class Communicator {
private:
    int mpiRank, worldSize;
    MPI_Comm mpiWorld;
public:
    Communicator(int argc, char** argv){
        mpiWorld = MPI_COMM_WORLD;
        CHECK_CALLMPI(MPI_Init(&argc, &argv));
        CHECK_CALLMPI(MPI_Comm_rank(mpiWorld, &mpiRank));
        CHECK_CALLMPI(MPI_Comm_size(mpiWorld, &worldSize));
        std::cout << "Rank at " << mpiRank << " in world ";
        std::cout << worldSize << std::endl;
    }

    ~Communicator(){
        CHECK_CALLMPI(MPI_Finalize());
    }

    int getRank() { return mpiRank; }
    int getWorldSize() { return worldSize; }
    MPI_Comm& getWorld() { return mpiWorld; }
};

#endif
