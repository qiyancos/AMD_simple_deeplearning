HIPCC=/opt/rocm/bin/hipcc

HIPBLAS_LIB=hipblas
MIOPEN_LIB=MIOpen
MPI_LIB=mpi

MPI_LIB_DIR=/mnt/lustre/share/platform/dep/openmpi-2.1.5/lib
HIPBLAS_LIB_DIR=/opt/rocm/hipblas/lib
INTELMPI_LIB_DIR=/mnt/lustre/share/intel64/lib

LOCAL_INC=include
LOCAL_LIB=$(PWD)/bin/liboperators.so
ROCM_INC=/opt/rocm/include
MIOPEN_INC=/opt/rocm/miopen/include
HIPBLAS_INC=/opt/rocm/hipblas/include

MPI_INC=/mnt/lustre/share/platform/dep/openmpi-2.1.5/include
INTELMPI_INC=/mnt/lustre/share/intel64/include

CXXFLAGS=-O3 --std=c++11 -I$(LOCAL_INC)
AMDTARGETS=--amdgpu-target=gfx900 --amdgpu-target=gfx906
AMDLIBS=-I$(ROCM_INC) -I$(MIOPEN_INC) -l$(MIOPEN_LIB) \
		-I$(HIPBLAS_INC) -l$(HIPBLAS_LIB) -L$(HIPBLAS_LIB_DIR)

LOCALLIBS=-loperators -L$(LOCAL_LIB)
MPILIBS=-I$(MPI_INC) -l$(MPI_LIB) -L$(MPI_LIB_DIR)
INTELMPILIBS=-I$(INTELMPI_INC) -L$(INTELMPI_LIB_DIR) -l$(MPI_LIB)

AMDCXXFLAGS=$(CXXFLAGS) $(AMDLIBS) $(AMDTARGETS)

HPPLIST=include/*.hpp
OPERATORLIST=operators/*.cpp

all: bin/test_avgpool_raw bin/test_patmpi bin/test_intelmpi bin/test_mpi \
	 bin/test_other bin/test_deconv_raw bin/test_deconv_beta_bug bin/test_vgg_bug \
	 $(PWD)/bin/liboperators.so bin/test_model_vgg_bug bin/test_hipblas_bug

$(PWD)/bin/liboperators.so: $(OPERATORLIST) $(HPPLIST)
	mkdir -p bin
	$(HIPCC) $(OPERATORLIST) -fPIC -shared -o bin/liboperators.so $(AMDCXXFLAGS)

bin/test_model_vgg_bug: test_model_vgg_bug.cpp $(PWD)/bin/liboperators.so $(HPPLIST)
	mkdir -p bin
	$(HIPCC) test_model_vgg_bug.cpp -o bin/test_model_vgg_bug $(AMDCXXFLAGS) $(LOCAL_LIB) $(MPILIBS)

bin/test_hipblas_bug: test_hipblas_bug.cpp $(PWD)/bin/liboperators.so $(HPPLIST)
	mkdir -p bin
	$(HIPCC) test_hipblas_bug.cpp -o bin/test_hipblas_bug $(AMDCXXFLAGS) $(LOCAL_LIB)

bin/test_patmpi: test_mpi.cpp $(HPPLIST)
	mkdir -p bin
	$(HIPCC) test_mpi.cpp -o bin/test_patmpi $(AMDCXXFLAGS) $(MPILIBS) -DUSE_COPY

bin/test_intelmpi: test_mpi.cpp $(HPPLIST)
	mkdir -p bin
	$(HIPCC) test_mpi.cpp -o bin/test_intelmpi $(AMDCXXFLAGS) $(INTELMPILIBS) -DUSE_COPY

bin/test_mpi: test_mpi.cpp $(HPPLIST)
	mkdir -p bin
	$(HIPCC) test_mpi.cpp -o bin/test_mpi $(AMDCXXFLAGS) $(MPILIBS)

bin/test_other: test_other.cpp $(HPPLIST)
	mkdir -p bin
	$(HIPCC) test_other.cpp -o bin/test_other $(AMDCXXFLAGS)

bin/test_deconv_raw: test_deconv_raw.cpp $(HPPLIST)
	mkdir -p bin
	$(HIPCC) test_deconv_raw.cpp -o bin/test_deconv_raw $(AMDCXXFLAGS)

bin/test_avgpool_raw: test_avgpool_raw.cpp $(HPPLIST)
	mkdir -p bin
	$(HIPCC) test_avgpool_raw.cpp -o bin/test_avgpool_raw $(AMDCXXFLAGS)

bin/test_deconv_beta_bug: test_deconv_beta_bug.cpp $(HPPLIST)
	mkdir -p bin
	$(HIPCC) test_deconv_beta_bug.cpp -o bin/test_deconv_beta_bug $(AMDCXXFLAGS)

bin/test_vgg_bug: test_vgg_bug.cpp $(HPPLIST)
	mkdir -p bin
	$(HIPCC) test_vgg_bug.cpp -o bin/test_vgg_bug $(AMDCXXFLAGS) $(MPILIBS)

clean:
	rm -rf bin
