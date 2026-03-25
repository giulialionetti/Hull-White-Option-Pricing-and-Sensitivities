NVCC  = $(shell which nvcc)
BIN   = bin
FLAGS = -arch=sm_70 -O3 -I./include

all: test

test: src/test.cu
	@mkdir -p $(BIN)
	$(NVCC) $(FLAGS) src/test.cu -o $(BIN)/test

run: test
	./$(BIN)/test

clean:
	rm -rf $(BIN)

.PHONY: all run clean