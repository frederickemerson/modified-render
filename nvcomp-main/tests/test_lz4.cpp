/*
 * Copyright (c) 2019-2020, NVIDIA CORPORATION. All rights reserved.
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

#include "nvcomp.hpp"
#include "nvcomp/lz4.hpp"

#include "catch.hpp"

#include <assert.h>
#include <stdlib.h>
#include <vector>

// Test GPU decompression with cascaded compression API //

using namespace std;
using namespace nvcomp;

#define CUDA_CHECK(cond)                                                       \
  do {                                                                         \
    cudaError_t err = cond;                                                    \
    REQUIRE(err == cudaSuccess);                                               \
  } while (false)

/******************************************************************************
 * HELPER FUNCTIONS ***********************************************************
 *****************************************************************************/

namespace
{

template <typename T>
std::vector<T> buildRuns(const size_t numRuns, const size_t runSize)
{
  std::vector<T> input;
  for (size_t i = 0; i < numRuns; i++) {
    for (size_t j = 0; j < runSize; j++) {
      input.push_back(static_cast<T>(i));
    }
  }

  return input;
}

template <typename T>
void test_lz4(const std::vector<T>& input, const size_t chunk_size = 1 << 16)
{
  // create GPU only input buffer
  T* d_in_data;
  const size_t in_bytes = sizeof(T) * input.size();
  CUDA_CHECK(cudaMalloc((void**)&d_in_data, in_bytes));
  CUDA_CHECK(
      cudaMemcpy(d_in_data, input.data(), in_bytes, cudaMemcpyHostToDevice));

  cudaStream_t stream;
  cudaStreamCreate(&stream);

  size_t comp_temp_bytes = 0;
  size_t comp_out_bytes = 0;
  void* d_comp_temp;
  void* d_comp_out;

  LZ4Compressor compressor(chunk_size);
  compressor.configure(in_bytes, &comp_temp_bytes, &comp_out_bytes);
  REQUIRE(comp_temp_bytes > 0);
  REQUIRE(comp_out_bytes > 0);

  // allocate temp buffer
  CUDA_CHECK(cudaMalloc(&d_comp_temp, comp_temp_bytes));

  // Allocate output buffer
  CUDA_CHECK(cudaMalloc(&d_comp_out, comp_out_bytes));

  size_t* comp_out_bytes_ptr;
  cudaMalloc((void**)&comp_out_bytes_ptr, sizeof(size_t));
  compressor.compress_async(
      d_in_data,
      in_bytes,
      d_comp_temp,
      comp_temp_bytes,
      d_comp_out,
      comp_out_bytes_ptr,
      stream);

  CUDA_CHECK(cudaStreamSynchronize(stream));
  CUDA_CHECK(cudaMemcpy(
      &comp_out_bytes,
      comp_out_bytes_ptr,
      sizeof(comp_out_bytes),
      cudaMemcpyDeviceToHost));
  cudaFree(comp_out_bytes_ptr);

  cudaFree(d_comp_temp);
  cudaFree(d_in_data);

  // Test to make sure copying the compressed file is ok
  void* copied = 0;
  CUDA_CHECK(cudaMalloc(&copied, comp_out_bytes));
  CUDA_CHECK(cudaMemcpy(copied, d_comp_out, comp_out_bytes, cudaMemcpyDeviceToDevice));
  cudaFree(d_comp_out);
  d_comp_out = copied;

  LZ4Decompressor decompressor;

  size_t decomp_temp_bytes;
  size_t decomp_out_bytes;
  decompressor.configure(
      d_comp_out,
      comp_out_bytes,
      &decomp_temp_bytes,
      &decomp_out_bytes,
      stream);

  void* d_decomp_temp;
  cudaMalloc(&d_decomp_temp, decomp_temp_bytes);

  T* out_ptr;
  cudaMalloc(&out_ptr, decomp_out_bytes);

  // make sure the data won't match input if not written to, so we can verify
  // correctness
  cudaMemset(out_ptr, 0, decomp_out_bytes);

  decompressor.decompress_async(
      d_comp_out,
      comp_out_bytes,
      d_decomp_temp,
      decomp_temp_bytes,
      out_ptr,
      decomp_out_bytes,
      stream);
  CUDA_CHECK(cudaStreamSynchronize(stream));

  // Copy result back to host
  std::vector<T> res(input.size());
  cudaMemcpy(
      &res[0], out_ptr, input.size() * sizeof(T), cudaMemcpyDeviceToHost);

  // Verify correctness
  REQUIRE(res == input);

  cudaFree(d_comp_out);
  cudaFree(out_ptr);
  cudaFree(d_decomp_temp);
}

} // namespace

/******************************************************************************
 * UNIT TESTS *****************************************************************
 *****************************************************************************/

TEST_CASE("comp/decomp LZ4-small", "[nvcomp]")
{
  using T = int;

  std::vector<T> input = {0, 2, 2, 3, 0, 0, 0, 0, 0, 3, 1, 1, 1, 1, 1, 2, 3, 3};

  test_lz4(input);
}

TEST_CASE("comp/decomp LZ4-1", "[nvcomp]")
{
  using T = int;

  const int num_elems = 500;
  std::vector<T> input;
  for (int i = 0; i < num_elems; ++i) {
    input.push_back(i >> 2);
  }

  test_lz4(input);
}

TEST_CASE("comp/decomp LZ4-all-small-sizes", "[nvcomp][small]")
{
  using T = uint8_t;

  for (int total = 1; total < 4096; ++total) {
    std::vector<T> input = buildRuns<T>(total, 1);
    test_lz4(input);
  }
}

TEST_CASE("comp/decomp LZ4-multichunk", "[nvcomp][large]")
{
  using T = int;

  for (int total = 10; total < (1 << 24); total = total * 2 + 7) {
    std::vector<T> input = buildRuns<T>(total, 10);
    test_lz4(input);
  }
}

TEST_CASE("comp/decomp LZ4-small-uint8", "[nvcomp][small]")
{
  using T = uint8_t;

  for (size_t num = 1; num < 1 << 18; num = num * 2 + 1) {
    std::vector<T> input = buildRuns<T>(num, 3);
    test_lz4(input);
  }
}

TEST_CASE("comp/decomp LZ4-small-uint16", "[nvcomp][small]")
{
  using T = uint16_t;

  for (size_t num = 1; num < 1 << 18; num = num * 2 + 1) {
    std::vector<T> input = buildRuns<T>(num, 3);
    test_lz4(input);
  }
}

TEST_CASE("comp/decomp LZ4-small-uint32", "[nvcomp][small]")
{
  using T = uint32_t;

  for (size_t num = 1; num < 1 << 18; num = num * 2 + 1) {
    std::vector<T> input = buildRuns<T>(num, 3);
    test_lz4(input);
  }
}

TEST_CASE("comp/decomp LZ4-small-uint64", "[nvcomp][small]")
{
  using T = uint64_t;

  for (size_t num = 1; num < 1 << 18; num = num * 2 + 1) {
    std::vector<T> input = buildRuns<T>(num, 3);
    test_lz4(input);
  }
}

TEST_CASE("comp/decomp LZ4-chunksizes-uint64", "[nvcomp][small]")
{
  using T = uint64_t;

  const size_t num = 2000000;

  std::vector<size_t> chunk_sizes{
      32768, 32769, 50000, 65535, 65536, 90103, 16777216};

  for (const size_t chunk : chunk_sizes) {
    std::vector<T> input = buildRuns<T>(num, 5);
    test_lz4(input, chunk);
  }
}
