# Torchbidings

We need to load a PyTorch model and make predictions, but since we are using Go, there's no native library to direclty run PyTorch models like in Python. To solve this, we can load the model in c++ using LibTorch (PyTorch's C++ API) and call the necessary functions from Go using `cmd/cgo`.

This was only possible because of this [excellent article](https://omkar.xyz/golibtorch/) that explains the process of integrating Go with LibTorch. Go read it first. The article didn't provide a way to make CUDA available, but I manage to do it with this [info](https://stackoverflow.com/questions/32589153/how-to-compile-cuda-source-with-go-languages-cgo).

## Requirements

- **Linux**
- **Cmake** (3.22.1)
- **Libtorch** (version 2.5.1 with minimum CUDA 12.4 support and cxx11 ABI enabled). Dowloand from:
  ```sh
  wget https://download.pytorch.org/libtorch/cu124/libtorch-cxx11-abi-shared-with-deps-2.5.1%2Bcu124.zip
  ```
- **Cuda Toolkit** (12.6) and **CUDA drivers**. Installation guide: [link](https://developer.nvidia.com/cuda-toolkit)
- **nlohmann/json** (3.11.3). Gihub page: [link](https://github.com/nlohmann/json)

## Build intructions

The shared object (`libclassifier.so`) must be built by linking both LibTorch and CUDA dependencies together.

1. **Create a `build` directory** and navigate to it:

   ```sh
   mkdir build
   cd build/
   ```

2. **Generate the build system files using CMake**:

   You need to specify the paths to CUDA and LibTorch. Ensure CUDACXX points to the correct CUDA compiler (nvcc) - In my case both are in `/usr/local`.

   ```sh
   CUDACXX=/usr/local/cuda-12.4/bin/nvcc cmake -DCMAKE_PREFIX_PATH=/usr/lib/libtorch ..
   ```

3. **Build the shared object** (`libclassifier.so`):

   ```sh
   cmake --build . --config Release
   ```

   This command will generate libclassifier.so in the build directory. This shared object contains the necessary C++ logic for loading the PyTorch model, applying MinMax scaling, and performing predictions.

## Go interface (`cmd/cgo`)

The Go code interacts with the shared object (`libclassifier.so`) using `cGo`, which allows Go programs to call C/C++ functions directly.

If you check the `classifier.go` file, you’ll notice that we are passing the `-L./build` flag to specify the directory where `libclassifier.so` is located, and linking it using `-lclassifier`:

```go
// #cgo LDFLAGS: -L./build -lclassifier -ltorch -ltorch_cpu -ltorch_cuda -lcudart -lc10
// #cgo CXXFLAGS: -std=c++17
// #cgo CFLAGS: -D_GLIBCXX_USE_CXX11_ABI=1
// #include <stdio.h>
// #include <stdlib.h>
// #include "classifier.h"
import "C"
```

## Running the code

To run the Go program successfully, you need to ensure that the following environment variables are set correctly. These variables specify the paths to LibTorch, CUDA, and the shared object (`libclassifier.so`):

```sh
export LIBRARY_PATH="/usr/local/libtorch/lib:/usr/local/cuda-12.4/lib64:${LIBRARY_PATH}" && \
export LD_LIBRARY_PATH="./build:/usr/local/libtorch/lib:/usr/local/cuda-12.4/lib64:${LD_LIBRARY_PATH}" && \
export C_INCLUDE_PATH="/usr/local/libtorch/include:/usr/local/libtorch/include/torch/csrc/api/include:${C_INCLUDE_PATH}" && \
export CPLUS_INCLUDE_PATH="/usr/local/libtorch/include:/usr/local/libtorch/include/torch/csrc/api/include:${CPLUS_INCLUDE_PATH}" && \
export CUDACXX=/usr/local/cuda-12.4/bin/nvcc && \
export LD_PRELOAD="/usr/local/cuda-12.4/lib64/libcudart.so /usr/local/libtorch/lib/libtorch_cuda.so" && \
go run cmd/*.go; \
```

- Instead of setting the environment variables manually, you can simply use the provided bash script [run.sh](../../../run.sh), which exports all necessary environment variables and runs the Go program:
