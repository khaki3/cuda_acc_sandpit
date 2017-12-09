CC   = pgcc
FLAG = -acc -Bstatic_pgi -O3 -Minfo=accel
OBJ  = laplace gemm nbody himeno

all : $(OBJ)

% : %.c
	$(CC) $(FLAG) -o $@ $<

clean :
	rm -f $(OBJ)
