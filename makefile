NVCC  = $(shell which nvcc)
BIN   = bin
FLAGS = -arch=sm_86 -O3 -I./include

SRCS  = src/pricing.cu src/MC.cu

all: pricing

pricing: $(SRCS)
	@mkdir -p $(BIN)
	$(NVCC) $(FLAGS) $(SRCS) -o $(BIN)/$@

run: pricing
	./$(BIN)/pricing

clean:
	rm -rf $(BIN)

.PHONY: all run clean