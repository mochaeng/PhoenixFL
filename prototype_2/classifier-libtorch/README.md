# Federated Classifier - Libtorch (DEMO)

This is a demo showing how to load a PyTorch model in C++ using LibTorch. I applied some of the concepts I learned here to integrate Go with LibTorch. [Here](../golang/internal/torchbidings/README.md) contains more explanations.

## Requirements

- **Cmake** (3.22.1)
- **Libtorch** (version 2.5.1 with minimum CUDA 12.4 support and cxx11 ABI enabled). Dowloand from:
  ```sh
  wget https://download.pytorch.org/libtorch/cu124/libtorch-cxx11-abi-shared-with-deps-2.5.1%2Bcu124.zip
  ```
- **Cuda Toolkit** (12.6) and **CUDA drivers**. Installation guide: [link](https://developer.nvidia.com/cuda-toolkit)

## Build instructions

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

3. **Build the executable**:

   ```sh
   cmake --build . --config Release
   ```

4. **Run it**:

   ```sh
   ./federated-classifier
   ```
