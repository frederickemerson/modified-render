# README for the nvCOMP library extensions

This library is mainly intended to be used with nvCOMP 
(https://github.com/NVIDIA/nvcomp), but can also be used independently. Please 
refer to the nvCOMP documentation and examples for usage instructions with 
nvCOMP and the `gdeflate.h`, `bitcomp.h` include files for 
documentation on standalone usage.

## Dependencies
The GDeflate CPU compression interface uses zlib under the hood. Ensure that
zlib dynamic libraries are available when running the provided benchmark
executable or if using the GDeflate CPU compression interfaces in your
application.

## Setting up extensions
The library is provided as a binary distribution compiled for a particular OS.
This distribution contains multiple directories corresponding to different CUDA
toolkit versions. Use the subdirectory corresponding to your CUDA toolkit 
version.

Ensure your NVIDIA driver version is compatible with the CUDA toolkit version
you choose. Refer to https://docs.nvidia.com/deploy/cuda-compatibility/index.html
for more details.

## Configuring nvCOMP extensions
To configure nvCOMP extensions, simply define the `NVCOMP_EXTS` variable
to allow CMake to find the library
```
git clone https://github.com/NVIDIA/nvcomp.git
cd nvcomp
mkdir build
cd build
cmake -DNVCOMP_EXTS_ROOT=/path/to/nvcomp_exts/${CUDA_VERSION} ..
make -j4
```
where `CUDA_VERSION` is the CUDA toolkit version you have (11.0 for example).

If you're building using CUDA 10 or less, you will need to specify a 
path to CUB on your system, of at least version 1.9.10.
```
cmake -DCUB_DIR=<path to cub repository>
```

## Running nvCOMP extensions
To get a quick sense of achievable compression ratios and decompression
throughput, use the available benchmark executable and pass in a file
```
./bin/benchmark_gdeflate /path/to/data/filename
./bin/benchmark_bitcomp /path/to/data/filename
```

You can also use the example from the nvCOMP library to do this. From the build
directory, run
```
./bin/gdeflate_cpu_example -f /path/to/data/filename
./bin/benchmark_gdeflate_chunked -f /path/to/data/filename
```
