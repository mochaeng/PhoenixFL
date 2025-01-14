#!/bin/bash

# set evironment variables for CUDA and LibTorch
export LIBRARY_PATH="/usr/local/libtorch/lib:/usr/local/cuda-12.4/lib64:${LIBRARY_PATH}"
export LD_LIBRARY_PATH="./build:/usr/local/libtorch/lib:/usr/local/cuda-12.4/lib64:${LD_LIBRARY_PATH}"
export C_INCLUDE_PATH="/usr/local/libtorch/include:/usr/local/libtorch/include/torch/csrc/api/include:${C_INCLUDE_PATH}"
export CPLUS_INCLUDE_PATH="/usr/local/libtorch/include:/usr/local/libtorch/include/torch/csrc/api/include:${CPLUS_INCLUDE_PATH}"
export CUDACXX=/usr/local/cuda-12.4/bin/nvcc
export LD_PRELOAD="/usr/local/cuda-12.4/lib64/libcudart.so /usr/local/libtorch/lib/libtorch_cuda.so"

rm -f data/workers/worker*

echo "Starting prototype 02 simulation..."
cd simulation/cmd/latency || exit
go build -o prot2-simu *.go
./prot2-simu -workers=0 -ispub=true -pub-interval=1ms -msg-limit=100_000 &
mb_pid=$!
cd ../../..

# preparation time
sleep 5

# start client packets sending for T seconds
kill -SIGUSR1 ${mb_pid}
sleep 120

# stop script
kill -SIGTERM ${mb_pid}
echo "prototype 02 script finished."
