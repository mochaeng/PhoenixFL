# Federated Classifier (Libtorch)

> Before running you need to configure your cuda and libtorch locations.

You need to install [nvidia-toolkit](https://developer.nvidia.com/cuda-toolkit).

```sh
mkdir build
cd build

CUDACXX=/path/to/cuda-xx/bin/nvcc cmake -DCMAKE_PREFIX_PATH=/path/to/libtorch ..
cmake --build . --config Release
./example-app
```

Example: `CUDACXX=/usr/local/cuda-12.4/bin/nvcc cmake -DCMAKE_PREFIX_PATH=/usr/lib/libtorch ..`
