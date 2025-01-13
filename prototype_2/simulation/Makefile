
.PHONY: run
run:
	export LIBRARY_PATH="/usr/local/libtorch/lib:/usr/local/cuda-12.4/lib64:${LIBRARY_PATH}" && \
	export LD_LIBRARY_PATH="./build:/usr/local/libtorch/lib:/usr/local/cuda-12.4/lib64:${LD_LIBRARY_PATH}" && \
	export C_INCLUDE_PATH="/usr/local/libtorch/include:/usr/local/libtorch/include/torch/csrc/api/include:${C_INCLUDE_PATH}" && \
	export CPLUS_INCLUDE_PATH="/usr/local/libtorch/include:/usr/local/libtorch/include/torch/csrc/api/include:${CPLUS_INCLUDE_PATH}" && \
	export CUDACXX=/usr/local/cuda-12.4/bin/nvcc && \
	export LD_PRELOAD="/usr/local/cuda-12.4/lib64/libcudart.so /usr/local/libtorch/lib/libtorch_cuda.so" && \
    go run cmd/*.go

.PHONY: tests
tests:
	export LIBRARY_PATH="/usr/local/libtorch/lib:/usr/local/cuda-12.4/lib64:${LIBRARY_PATH}" && \
	export LD_LIBRARY_PATH="./build:/usr/local/libtorch/lib:/usr/local/cuda-12.4/lib64:${LD_LIBRARY_PATH}" && \
	export C_INCLUDE_PATH="/usr/local/libtorch/include:/usr/local/libtorch/include/torch/csrc/api/include:${C_INCLUDE_PATH}" && \
	export CPLUS_INCLUDE_PATH="/usr/local/libtorch/include:/usr/local/libtorch/include/torch/csrc/api/include:${CPLUS_INCLUDE_PATH}" && \
	export CUDACXX=/usr/local/cuda-12.4/bin/nvcc && \
	export LD_PRELOAD="/usr/local/cuda-12.4/lib64/libcudart.so /usr/local/libtorch/lib/libtorch_cuda.so" && \
    go test -race -v ./...
