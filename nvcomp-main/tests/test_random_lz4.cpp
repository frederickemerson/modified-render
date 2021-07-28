/*
 * Copyright (c) 2018-2020, NVIDIA CORPORATION. All rights reserved.
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

#define CATCH_CONFIG_MAIN

#ifndef VERBOSE
#define VERBOSE 0
#endif

#include "test_common.h"

// Test method that takes an input data, compresses it (on the CPU),
// decompresses it on the GPU, and verifies it is correct.
// Uses LZ4 Compression
template <typename T>
void test_lz4(const std::vector<T>& data, size_t /*chunk_size*/)
{
  const nvcompType_t type = nvcomp::TypeOf<T>();

  size_t chunk_size = 1 << 16;

#if VERBOSE > 1
  // dump input data
  std::cout << "Input" << std::endl;
  for (size_t i = 0; i < data.size(); i++)
    std::cout << data[i] << " ";
  std::cout << std::endl;
#endif

  // these two items will be the only forms of communication between
  // compression and decompression
  void* d_comp_out = nullptr;
  size_t comp_out_bytes = 0;
  {
    // this block handles compression, and we scope it to ensure only
    // serialized metadata and compressed data, are the only things passed
    // between compression and decopmression
    std::cout << "----------" << std::endl;
    std::cout << "uncompressed (B): " << data.size() * sizeof(T) << std::endl;

    // create GPU only input buffer
    void* d_in_data;
    const size_t in_bytes = sizeof(T) * data.size();
    CUDA_CHECK(cudaMalloc(&d_in_data, in_bytes));
    CUDA_CHECK(
        cudaMemcpy(d_in_data, data.data(), in_bytes, cudaMemcpyHostToDevice));

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    nvcompError_t status;

    LZ4Compressor compressor(chunk_size);
    size_t comp_temp_bytes;
    compressor.configure(in_bytes, &comp_temp_bytes, &comp_out_bytes);

    void* d_comp_temp;
    CUDA_CHECK(cudaMalloc(&d_comp_temp, comp_temp_bytes));
    CUDA_CHECK(cudaMalloc(&d_comp_out, comp_out_bytes));

    size_t* comp_out_bytes_ptr;
    CUDA_CHECK(cudaMallocHost(
        (void**)&comp_out_bytes_ptr, sizeof(*comp_out_bytes_ptr)));

    compressor.compress_async(
        d_in_data,
        in_bytes,
        d_comp_temp,
        comp_temp_bytes,
        d_comp_out,
        comp_out_bytes_ptr,
        stream);

    CUDA_CHECK(cudaStreamSynchronize(stream));
    comp_out_bytes = *comp_out_bytes_ptr;

    cudaFree(d_comp_temp);
    cudaFree(d_in_data);
    cudaStreamDestroy(stream);

    std::cout << "comp_size: " << comp_out_bytes
              << ", compressed ratio: " << std::fixed << std::setprecision(2)
              << (double)in_bytes / comp_out_bytes << std::endl;
  }

  {
    // this block handles decompression, and we scope it to ensure only
    // serialized metadata and compressed data, are the only things passed
    // between compression and decopmression
    //

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    LZ4Decompressor decompressor;
    size_t temp_bytes;
    size_t decomp_out_bytes;
    decompressor.configure(
        d_comp_out, comp_out_bytes, &temp_bytes, &decomp_out_bytes, stream);

    void* temp_ptr;
    cudaMalloc(&temp_ptr, temp_bytes);
    T* out_ptr = NULL;
    cudaMalloc((void**)&out_ptr, decomp_out_bytes);

    auto start = std::chrono::steady_clock::now();

    decompressor.decompress_async(
        d_comp_out,
        comp_out_bytes,
        temp_ptr,
        temp_bytes,
        out_ptr,
        decomp_out_bytes,
        stream);

    CUDA_CHECK(cudaStreamSynchronize(stream));

    // stop timing and the profiler
    auto end = std::chrono::steady_clock::now();
    std::cout << "throughput (GB/s): " << gbs(start, end, decomp_out_bytes)
              << std::endl;

    cudaStreamDestroy(stream);
    cudaFree(d_comp_out);
    cudaFree(temp_ptr);

    std::vector<T> res(decomp_out_bytes / sizeof(T));
    cudaMemcpy(&res[0], out_ptr, decomp_out_bytes, cudaMemcpyDeviceToHost);

#if VERBOSE > 1
    // dump output data
    std::cout << "Output" << std::endl;
    for (size_t i = 0; i < data.size(); i++)
      std::cout << ((T*)out_ptr)[i] << " ";
    std::cout << std::endl;
#endif

    REQUIRE(res == data);
  }
}

template <typename T>
void test_random_lz4(
    int max_val,
    int max_run,
    size_t chunk_size)
{
  // generate random data
  std::vector<T> data;
  int seed = (max_val ^ max_run ^ chunk_size);
  random_runs(data, (T)max_val, (T)max_run, seed);

  test_lz4<T>(data, chunk_size);
}

// int
TEST_CASE("small-LZ4", "[small]")
{
  test_random_lz4<int>(10, 10, 10000);
}
TEST_CASE("medium-LZ4", "[small]")
{
  test_random_lz4<int>(10000, 10, 100000);
}

TEST_CASE("large-LZ4", "[large][bp]")
{
  test_random_lz4<int>(10000, 1000, 10000000);
}



// long long
TEST_CASE("small-LZ4-ll", "[small]")
{
  test_random_lz4<int64_t>(10, 10, 10000);
}
TEST_CASE("large-LZ4-ll", "[large]")
{
  test_random_lz4<int64_t>(10000, 1000, 10000000);
}
