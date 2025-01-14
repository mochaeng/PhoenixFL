SHELL := /bin/bash


CUDA_PATH := /usr/local/cuda-12.4
TORCH_PATH := /usr/local/libtorch

# lib and include paths
CUDA_LIB := $(CUDA_PATH)/lib64
TORCH_LIB := $(TORCH_PATH)/lib
TORCH_INCLUDE := $(TORCH_PATH)/include
TORCH_API_INCLUDE := $(TORCH_INCLUDE)/torch/csrc/api/include

export LIBRARY_PATH := $(TORCH_LIB):$(CUDA_LIB):${LIBRARY_PATH}
export LD_LIBRARY_PATH := ./build:$(TORCH_LIB):$(CUDA_LIB):${LD_LIBRARY_PATH}
export C_INCLUDE_PATH := $(TORCH_INCLUDE):$(TORCH_API_INCLUDE):${C_INCLUDE_PATH}
export CPLUS_INCLUDE_PATH := $(TORCH_INCLUDE):$(TORCH_API_INCLUDE):${CPLUS_INCLUDE_PATH}
export CUDACXX := $(CUDA_PATH)/bin/nvcc
export LD_PRELOAD := $(CUDA_LIB)/libcudart.so $(TORCH_LIB)/libtorch_cuda.so


WORKERS ?= 0
PUB_INTERVAL ?= 1ms
MSG_LIMIT ?= 100000
PREP_TIME ?= 5
SIM_TIME ?= 40

.PHONY: clean build-latency run-latency simulation-latency

clean:
	rm -f data/workers/worker*
	rm -f simulation/cmd/latency/prot2-simu

build-latency:
	cd simulation/cmd/latency && go build -o prot2-simu *.go

run-latency: build-latency
	cd simulation/cmd/latency && \
	( ./prot2-simu -workers=$(WORKERS) -ispub=true -pub-interval=$(PUB_INTERVAL) \
	  -msg-limit=$(MSG_LIMIT) & \
	  mb_pid=$$! && echo $$mb_pid && sleep $(PREP_TIME) && \
	  kill -SIGUSR1 $$mb_pid && sleep $(SIM_TIME) && \
	  kill -SIGTERM $$mb_pid )

simulation-latency: clean run-latency
	@echo "Prototype 02 simulation finished."


# Environment variables for CUDA and LibTorch
# export LIBRARY_PATH := /usr/local/libtorch/lib:/usr/local/cuda-12.4/lib64:${LIBRARY_PATH}
# export LD_LIBRARY_PATH := ./build:/usr/local/libtorch/lib:/usr/local/cuda-12.4/lib64:${LD_LIBRARY_PATH}
# export C_INCLUDE_PATH := /usr/local/libtorch/include:/usr/local/libtorch/include/torch/csrc/api/include:${C_INCLUDE_PATH}
# export CPLUS_INCLUDE_PATH := /usr/local/libtorch/include:/usr/local/libtorch/include/torch/csrc/api/include:${CPLUS_INCLUDE_PATH}
# export CUDACXX := /usr/local/cuda-12.4/bin/nvcc
# export LD_PRELOAD := /usr/local/cuda-12.4/lib64/libcudart.so /usr/local/libtorch/lib/libtorch_cuda.so

# .PHONY: run
# run:
#     go run cmd/*.go

# .PHONY: tests
# tests:
# 	export LIBRARY_PATH="/usr/local/libtorch/lib:/usr/local/cuda-12.4/lib64:${LIBRARY_PATH}" && \
# 	export LD_LIBRARY_PATH="./build:/usr/local/libtorch/lib:/usr/local/cuda-12.4/lib64:${LD_LIBRARY_PATH}" && \
# 	export C_INCLUDE_PATH="/usr/local/libtorch/include:/usr/local/libtorch/include/torch/csrc/api/include:${C_INCLUDE_PATH}" && \
# 	export CPLUS_INCLUDE_PATH="/usr/local/libtorch/include:/usr/local/libtorch/include/torch/csrc/api/include:${CPLUS_INCLUDE_PATH}" && \
# 	export CUDACXX=/usr/local/cuda-12.4/bin/nvcc && \
# 	export LD_PRELOAD="/usr/local/cuda-12.4/lib64/libcudart.so /usr/local/libtorch/lib/libtorch_cuda.so" && \
#     go test -race -v ./...
