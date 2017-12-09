NVCC   = nvcc
FLAG   = -O3 --gpu-architecture=compute_30
OBJ    = hello gemm nbody laplace himeno
HEADER = common.h matrix.h

all : $(OBJ)

% : %.cu $(HEADER)
	$(NVCC) $(FLAG) -o $@ $<

clean :
	rm -f $(OBJ)
