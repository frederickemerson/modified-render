/*
 * Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#ifndef VERBOSE
#define VERBOSE 0
#endif

#include "nvcomp/lz4.hpp"

#include "benchmark_common.h"

#include <string.h>
#include <string>
#include <vector>

using namespace nvcomp;

namespace
{

constexpr const size_t CHUNK_SIZE = 1 << 16;

void print_usage()
{
  printf("Usage: benchmark_binary [OPTIONS]\n");
  printf("  %-35s GPU device number (default 0)\n", "-g, --gpu");
  exit(1);
}

// Benchmark performance from the binary data file fname
void run_benchmark(const std::vector<uint8_t>& data)
{
  const size_t num_bytes = data.size();

  // Make sure dataset fits on GPU to benchmark total compression
  size_t freeMem;
  size_t totalMem;
  cudaMemGetInfo(&freeMem, &totalMem);
  if (freeMem < num_bytes) {
    std::cout << "Insufficient GPU memory to perform compression." << std::endl;
    exit(1);
  }

  const size_t num_chunks = roundUpDiv(num_bytes, CHUNK_SIZE);

  std::cout << "----------" << std::endl;
  std::cout << "uncompressed (B): " << num_bytes << std::endl;
  std::cout << "chunks " << num_chunks << std::endl;

  uint8_t* d_in_data;
  CUDA_CHECK(cudaMalloc((void**)&d_in_data, num_bytes));
  CUDA_CHECK(
      cudaMemcpy(d_in_data, data.data(), num_bytes, cudaMemcpyHostToDevice));

  LZ4Compressor compressor(CHUNK_SIZE);

  size_t comp_temp_bytes;
  size_t comp_out_bytes;
  compressor.configure(num_bytes, &comp_temp_bytes, &comp_out_bytes);
  benchmark_assert(
      comp_out_bytes > 0, "Output size must be greater than zero.");

  cudaStream_t stream;
  cudaStreamCreate(&stream);

  // Get temp size needed for compression

  // Allocate temp workspace
  void* d_comp_temp;
  CUDA_CHECK(cudaMalloc(&d_comp_temp, comp_temp_bytes));

  // Allocate compressed output buffer
  void* d_comp_out;
  CUDA_CHECK(cudaMalloc(&d_comp_out, comp_out_bytes));

  // Launch compression
  size_t* d_comp_out_bytes;
  CUDA_CHECK(
      cudaMallocHost((void**)&d_comp_out_bytes, sizeof(*d_comp_out_bytes)));

  auto start = std::chrono::steady_clock::now();
  compressor.compress_async(
      d_in_data,
      num_bytes,
      d_comp_temp,
      comp_temp_bytes,
      d_comp_out,
      d_comp_out_bytes,
      stream);
  CUDA_CHECK(cudaStreamSynchronize(stream));
  auto end = std::chrono::steady_clock::now();

  comp_out_bytes = *d_comp_out_bytes;
  CUDA_CHECK(cudaFreeHost(d_comp_out_bytes));

  CUDA_CHECK(cudaFree(d_comp_temp));
  CUDA_CHECK(cudaFree(d_in_data));

  std::cout << "comp_size: " << comp_out_bytes
            << ", compressed ratio: " << std::fixed << std::setprecision(2)
            << (double)num_bytes / comp_out_bytes << std::endl;
  std::cout << "compression throughput (GB/s): " << gbs(start, end, num_bytes)
            << std::endl;

  // get metadata from compressed data on GPU
  LZ4Decompressor decompressor;

  size_t decomp_temp_bytes;
  size_t decomp_bytes;
  decompressor.configure(
      d_comp_out, comp_out_bytes, &decomp_temp_bytes, &decomp_bytes, stream);

  // allocate temp buffer
  void* d_decomp_temp;
  CUDA_CHECK(cudaMalloc(
      &d_decomp_temp, decomp_temp_bytes)); // also can use RMM_ALLOC instead

  // allocate output buffer
  uint8_t* decomp_out_ptr;
  CUDA_CHECK(cudaMalloc(
      (void**)&decomp_out_ptr, decomp_bytes)); // also can use RMM_ALLOC instead

  start = std::chrono::steady_clock::now();

  // execute decompression (asynchronous)
  decompressor.decompress_async(
      d_comp_out,
      comp_out_bytes,
      d_decomp_temp,
      decomp_temp_bytes,
      decomp_out_ptr,
      decomp_bytes,
      stream);

  CUDA_CHECK(cudaStreamSynchronize(stream));

  // stop timing and the profiler
  end = std::chrono::steady_clock::now();
  std::cout << "decompression throughput (GB/s): "
            << gbs(start, end, decomp_bytes) << std::endl;

  CUDA_CHECK(cudaStreamDestroy(stream));
  CUDA_CHECK(cudaFree(d_decomp_temp));
  CUDA_CHECK(cudaFree(d_comp_out));

  benchmark_assert(
      decomp_bytes == num_bytes, "Decompressed result incorrect size.");

  std::vector<uint8_t> res(num_bytes);
  CUDA_CHECK(cudaMemcpy(
      res.data(), decomp_out_ptr, num_bytes, cudaMemcpyDeviceToHost));

  benchmark_assert(res == data, "Decompressed data does not match input.");
}

void run_tests(std::mt19937& rng)
{
  // test all zeros
  for (size_t b = 0; b < 14; ++b) {
    run_benchmark(gen_data(0, CHUNK_SIZE << b, rng));
  }

  // test random bytes
  for (size_t b = 0; b < 14; ++b) {
    run_benchmark(gen_data(255, CHUNK_SIZE << b, rng));
  }
}

} // namespace

int main(int argc, char* argv[])
{
  int gpu_num = 0;

  // Parse command-line arguments
  char** argv_end = argv + argc;
  argv += 1;
  while (argv != argv_end) {
    char* arg = *argv++;
    if (strcmp(arg, "--help") == 0 || strcmp(arg, "-?") == 0) {
      print_usage();
      return 1;
    }

    // all arguments below require at least a second value in argv
    if (argv >= argv_end) {
      print_usage();
      return 1;
    }

    char* optarg = *argv++;
    if (strcmp(arg, "--gpu") == 0 || strcmp(arg, "-g") == 0) {
      gpu_num = atoi(optarg);
      continue;
    }
    print_usage();
    return 1;
  }
  CUDA_CHECK(cudaSetDevice(gpu_num));

  std::mt19937 rng(0);

  run_tests(rng);

  return 0;
}
