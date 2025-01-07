# Federated Classifier (Libtorch)

> Before running you need to configure your cuda and libtorch locations.

```sh
mkdir build
cd build

CUDACXX=/path/to/cuda-xx/bin/nvcc cmake -DCMAKE_PREFIX_PATH=/path/to/libtorch ..
cmake --build . --config Release
./example-app
```

Example: `CUDACXX=/usr/local/cuda-12.4/bin/nvcc cmake -DCMAKE_PREFIX_PATH=/home/campos/cpp_libs/libtorch ..`
