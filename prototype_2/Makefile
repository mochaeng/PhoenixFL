SHELL := /bin/bash

# [change] this two variables based on your own paths
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


MODEL_PATH ?= "../../../data/fedmedian_model.pt"
CSV_PATH ?= "../../../data/10_000-raw-packets.csv"
WORKERS ?= 5

IS_PUB ?= true
MSG_LIMIT ?= 24_000
PUB_INTERVAL ?= 5ms

WORKER_IDLE_TIMEOUT ?= 20s

PREP_TIME ?= 5
SIM_TIME ?= 120

.PHONY: clean build-latency build-throughput run-latency run-throughput simulation-latency \
		simulation-throughput clean-throughput run-normalflow build-normalflow

clean:
	rm -f data/workers/worker*
	rm -f simulation/cmd/latency/prot2-simu
	rm -f simulation/cmd/throughput/prot2-simu
	rm -f simulation/cmd/normalflow/prot2-simu

clean-throughput: clean
	find data/workers -type f -name '*-simulation-throughput*' \
	                  -regex '.*/[0-9]+-[0-9]+-simulation-throughput.*' \
					  -exec rm -f {} +

build-latency:
	cd simulation/cmd/latency && go build -o prot2-simu *.go

build-throughput:
	cd simulation/cmd/throughput && go build -o prot2-simu *.go

build-normalflow:
	cd simulation/cmd/normalflow && go build -o prot2-simu main.go

run-latency: build-latency
	cd simulation/cmd/latency && \
	( ./prot2-simu -workers=$(WORKERS) \
	               -ispub=$(IS_PUB) \
	               -pub-interval=$(PUB_INTERVAL) \
	               -msg-limit=$(MSG_LIMIT) \
				   -model=$(MODEL_PATH) \
				   -csv=$(CSV_PATH) & \
	  mb_pid=$$! && echo $$mb_pid && sleep $(PREP_TIME) && \
	  kill -SIGUSR1 $$mb_pid && sleep $(SIM_TIME) && \
	  kill -SIGTERM $$mb_pid && wait $$mb_pid )

run-throughput: build-throughput
	cd simulation/cmd/throughput && \
	( ./prot2-simu -workers=$(WORKERS) \
	               -msg-limit=$(MSG_LIMIT) \
	               -idle-timeout=$(WORKER_IDLE_TIMEOUT) \
				   -model=$(MODEL_PATH) \
				   -csv=$(CSV_PATH) )

run-normalflow: build-normalflow
	cd simulation/cmd/normalflow && \
	./prot2-simu -workers=$(WORKERS) \
	             -pub-interval=$(PUB_INTERVAL) \
	             -model=$(MODEL_PATH) \
				 -csv=$(CSV_PATH)

simulation-latency: clean run-latency
	@echo "Prototype 02 latency simulation has finished."

simulation-throughput: clean run-throughput
	@echo "Prototype 02 throughput simulation has finished."

simulation-normalflow: clean run-normalflow
	@echo "Normalflow simulation has finished."
