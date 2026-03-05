NVCC  = $(shell which nvcc)
BIN   = bin
FLAGS = -arch=sm_70 -O3 -I./include

all:  MC


MC: src/MC.cu
	@mkdir -p $(BIN)
	$(NVCC) $(FLAGS) src/MC.cu -o $(BIN)/MC



run_MC: MC
	./$(BIN)/MC

clean:
	rm -rf $(BIN)

.PHONY: all run_MC clean