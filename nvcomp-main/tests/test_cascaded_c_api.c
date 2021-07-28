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

#include "nvcomp.h"
#include "nvcomp/cascaded.h"

#include "cuda_runtime.h"

#include <assert.h>
#include <stddef.h>
#include <stdio.h>

// Test GPU decompression with cascaded compression API //

#define REQUIRE(a)                                                             \
  do {                                                                         \
    if (!(a)) {                                                                \
      printf("Check " #a " at %d failed.\n", __LINE__);                        \
      return 0;                                                                \
    }                                                                          \
  } while (0)

#define CUDA_CHECK(func)                                                       \
  do {                                                                         \
    cudaError_t rt = (func);                                                   \
    if (rt != cudaSuccess) {                                                   \
      printf(                                                                  \
          "API call failure \"" #func "\" with %d at " __FILE__ ":%d\n",       \
          (int)rt,                                                             \
          __LINE__);                                                           \
      return 0;                                                                \
    }                                                                          \
  } while (0)

static int check_cascaded(const nvcompCascadedFormatOpts comp_opts)
{
  typedef int T;
  const nvcompType_t type = NVCOMP_TYPE_INT;

  const size_t input_size = 16;
  T input[16] = {0, 2, 2, 3, 0, 0, 0, 0, 0, 3, 1, 1, 1, 1, 1, 1};

  // create GPU only input buffer
  void* d_in_data;
  const size_t in_bytes = sizeof(T) * input_size;
  CUDA_CHECK(cudaMalloc(&d_in_data, in_bytes));
  CUDA_CHECK(cudaMemcpy(d_in_data, input, in_bytes, cudaMemcpyHostToDevice));

  cudaStream_t stream;
  cudaStreamCreate(&stream);

  nvcompError_t status;

  // Compress on the GPU
  size_t comp_temp_bytes;
  size_t comp_out_bytes;
  size_t metadata_bytes;
  status = nvcompCascadedCompressConfigure(
      &comp_opts,
      type,
      in_bytes,
      &metadata_bytes,
      &comp_temp_bytes,
      &comp_out_bytes);
  REQUIRE(status == nvcompSuccess);

  void* d_comp_temp;
  void* d_comp_out;
  CUDA_CHECK(cudaMalloc(&d_comp_temp, comp_temp_bytes));
  CUDA_CHECK(cudaMalloc(&d_comp_out, comp_out_bytes));

  size_t* d_comp_out_bytes;
  CUDA_CHECK(cudaMalloc((void**)&d_comp_out_bytes, sizeof(*d_comp_out_bytes)));
  CUDA_CHECK(cudaMemcpy(
      d_comp_out_bytes,
      &comp_out_bytes,
      sizeof(*d_comp_out_bytes),
      cudaMemcpyHostToDevice));

  status = nvcompCascadedCompressAsync(
      &comp_opts,
      type,
      d_in_data,
      in_bytes,
      d_comp_temp,
      comp_temp_bytes,
      d_comp_out,
      d_comp_out_bytes,
      stream);
  REQUIRE(status == nvcompSuccess);
  CUDA_CHECK(cudaStreamSynchronize(stream));

  CUDA_CHECK(cudaMemcpy(
      &comp_out_bytes,
      d_comp_out_bytes,
      sizeof(comp_out_bytes),
      cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaFree(d_comp_out_bytes));
  CUDA_CHECK(cudaFree(d_comp_temp));
  CUDA_CHECK(cudaFree(d_in_data));

  // get temp and output size
  size_t temp_bytes;
  size_t output_bytes;
  void* metadata_ptr = NULL;

  status = nvcompCascadedDecompressConfigure(
      d_comp_out,
      comp_out_bytes,
      &metadata_ptr,
      &metadata_bytes,
      &temp_bytes,
      &output_bytes,
      stream);
  REQUIRE(status == nvcompSuccess);

  // allocate temp buffer
  void* temp_ptr;
  CUDA_CHECK(cudaMalloc(&temp_ptr, temp_bytes));
  // allocate output buffer
  void* out_ptr;
  CUDA_CHECK(cudaMalloc(&out_ptr, output_bytes));

  // execute decompression (asynchronous)
  status = nvcompCascadedDecompressAsync(
      d_comp_out,
      comp_out_bytes,
      metadata_ptr,
      metadata_bytes,
      temp_ptr,
      temp_bytes,
      out_ptr,
      output_bytes,
      stream);
  REQUIRE(status == nvcompSuccess);

  CUDA_CHECK(cudaDeviceSynchronize());

  nvcompCascadedDestroyMetadata(metadata_ptr);

  // Copy result back to host
  int res[16];
  cudaMemcpy(res, out_ptr, output_bytes, cudaMemcpyDeviceToHost);

  CUDA_CHECK(cudaFree(temp_ptr));
  CUDA_CHECK(cudaFree(d_comp_out));

  // Verify correctness
  for (size_t i = 0; i < input_size; ++i) {
    REQUIRE(res[i] == input[i]);
  }

  return 1;
}

int test_rle_delta(void)
{
  nvcompCascadedFormatOpts comp_opts;
  comp_opts.num_RLEs = 1;
  comp_opts.num_deltas = 1;
  comp_opts.use_bp = 0;

  return check_cascaded(comp_opts);
}

int test_rle_delta_bp(void)
{
  nvcompCascadedFormatOpts comp_opts;
  comp_opts.num_RLEs = 1;
  comp_opts.num_deltas = 1;
  comp_opts.use_bp = 1;

  return check_cascaded(comp_opts);
}

int test_ones_init_data(void)
{
  typedef int T;
  const nvcompType_t type = NVCOMP_TYPE_INT;

  const size_t input_size = 12;
  const int input[12] = {0, 2, 2, 3, 0, 0, 3, 1, 1, 1, 1, 1};

  for (int packing = 0; packing <= 1; ++packing) {
    for (int Delta = 0; Delta <= 5; ++Delta) {
      for (int RLE = 0; RLE <= 5; ++RLE) {
        if (RLE + Delta + packing == 0) {
          // don't bother if there is no compression
          continue;
        }

        // create GPU only input buffer
        void* d_in_data;
        const size_t in_bytes = sizeof(T) * input_size;
        CUDA_CHECK(cudaMalloc(&d_in_data, in_bytes));
        CUDA_CHECK(
            cudaMemcpy(d_in_data, input, in_bytes, cudaMemcpyHostToDevice));

        nvcompCascadedFormatOpts comp_opts;
        comp_opts.num_RLEs = RLE;
        comp_opts.num_deltas = Delta;
        comp_opts.use_bp = packing;

        cudaStream_t stream;
        CUDA_CHECK(cudaStreamCreate(&stream));

        nvcompError_t status;

        // Compress on the GPU
        size_t comp_temp_bytes;
        size_t comp_out_bytes;
        size_t metadata_bytes;
        status = nvcompCascadedCompressConfigure(
            &comp_opts,
            type,
            in_bytes,
            &metadata_bytes,
            &comp_temp_bytes,
            &comp_out_bytes);
        REQUIRE(status == nvcompSuccess);

        void* d_comp_temp;
        void* d_comp_out;
        CUDA_CHECK(cudaMalloc(&d_comp_temp, comp_temp_bytes));
        CUDA_CHECK(cudaMalloc(&d_comp_out, comp_out_bytes));

        size_t* d_comp_out_bytes;
        CUDA_CHECK(
            cudaMalloc((void**)&d_comp_out_bytes, sizeof(*d_comp_out_bytes)));

        status = nvcompCascadedCompressAsync(
            &comp_opts,
            type,
            d_in_data,
            in_bytes,
            d_comp_temp,
            comp_temp_bytes,
            d_comp_out,
            d_comp_out_bytes,
            stream);
        REQUIRE(status == nvcompSuccess);
        CUDA_CHECK(cudaStreamSynchronize(stream));
        CUDA_CHECK(cudaMemcpy(
            &comp_out_bytes,
            d_comp_out_bytes,
            sizeof(comp_out_bytes),
            cudaMemcpyDeviceToHost));

        CUDA_CHECK(cudaFree(d_comp_out_bytes));
        CUDA_CHECK(cudaFree(d_comp_temp));
        CUDA_CHECK(cudaFree(d_in_data));

        // Perform Decompression using existing Metadata

        // get temp and output size
        size_t temp_bytes;
        size_t output_bytes;
        void* metadata_ptr = NULL;

        status = nvcompCascadedDecompressConfigure(
            d_comp_out,
            comp_out_bytes,
            &metadata_ptr,
            &metadata_bytes,
            &temp_bytes,
            &output_bytes,
            stream);
        REQUIRE(status == nvcompSuccess);

        // allocate temp buffer
        void* temp_ptr;
        CUDA_CHECK(cudaMalloc(&temp_ptr, temp_bytes));
        // allocate output buffer
        void* out_ptr;
        CUDA_CHECK(cudaMalloc(&out_ptr, output_bytes));

        // execute decompression (asynchronous)
        status = nvcompCascadedDecompressAsync(
            d_comp_out,
            comp_out_bytes,
            metadata_ptr,
            metadata_bytes,
            temp_ptr,
            temp_bytes,
            out_ptr,
            output_bytes,
            stream);
        REQUIRE(status == nvcompSuccess);

        CUDA_CHECK(cudaStreamSynchronize(stream));

        // Destory the metadata object and free memory
        nvcompCascadedDestroyMetadata(metadata_ptr);

        // Copy result back to host
        int res[12];
        CUDA_CHECK(
            cudaMemcpy(res, out_ptr, output_bytes, cudaMemcpyDeviceToHost));

        CUDA_CHECK(cudaFree(temp_ptr));
        CUDA_CHECK(cudaFree(d_comp_out));

        // Verify result
        for (size_t i = 0; i < input_size; ++i) {
          REQUIRE(res[i] == input[i]);
        }
      }
    }
  }

  return 1;
}

// TODO: to be removed in a future release when we don't need backward
// compatibility with 2.0.0.
static int test_cascaded_backward_compatibility(void)
{
  typedef int T;
  const nvcompType_t type = NVCOMP_TYPE_INT;

  const size_t input_size = 16;
  T input[16] = {0, 2, 2, 3, 0, 0, 0, 0, 0, 3, 1, 1, 1, 1, 1, 1};

  nvcompCascadedFormatOpts comp_opts;
  comp_opts.num_RLEs = 1;
  comp_opts.num_deltas = 1;
  comp_opts.use_bp = 0;

  // create GPU only input buffer
  void* d_in_data;
  const size_t in_bytes = sizeof(T) * input_size;
  CUDA_CHECK(cudaMalloc(&d_in_data, in_bytes));
  CUDA_CHECK(cudaMemcpy(d_in_data, input, in_bytes, cudaMemcpyHostToDevice));

  cudaStream_t stream;
  CUDA_CHECK(cudaStreamCreate(&stream));

  nvcompError_t status;

  // Compress on the GPU
  size_t comp_temp_bytes;
  size_t comp_out_bytes;
  size_t metadata_bytes;
  status = nvcompCascadedCompressConfigure(
      &comp_opts,
      type,
      in_bytes,
      &metadata_bytes,
      &comp_temp_bytes,
      &comp_out_bytes);
  REQUIRE(status == nvcompSuccess);

  void* d_comp_temp;
  void* d_comp_out;
  CUDA_CHECK(cudaMalloc(&d_comp_temp, comp_temp_bytes));
  CUDA_CHECK(cudaMalloc(&d_comp_out, comp_out_bytes));

  // NOTE: we are passingin comp_out_bytes as unregistered host memory, to test
  // that we are backward compatible with original 2.0.0 implementation, which
  // accepted it (a bug). We will remove this test (and compatibility) in a
  // futrue release.
  status = nvcompCascadedCompressAsync(
      &comp_opts,
      type,
      d_in_data,
      in_bytes,
      d_comp_temp,
      comp_temp_bytes,
      d_comp_out,
      &comp_out_bytes,
      stream);
  REQUIRE(status == nvcompSuccess);
  CUDA_CHECK(cudaStreamSynchronize(stream));

  CUDA_CHECK(cudaFree(d_comp_temp));
  CUDA_CHECK(cudaFree(d_in_data));

  // get temp and output size
  size_t temp_bytes;
  size_t output_bytes;
  void* metadata_ptr = NULL;

  status = nvcompCascadedDecompressConfigure(
      d_comp_out,
      comp_out_bytes,
      &metadata_ptr,
      &metadata_bytes,
      &temp_bytes,
      &output_bytes,
      stream);
  REQUIRE(status == nvcompSuccess);

  // allocate temp buffer
  void* temp_ptr;
  CUDA_CHECK(cudaMalloc(&temp_ptr, temp_bytes));
  // allocate output buffer
  void* out_ptr;
  CUDA_CHECK(cudaMalloc(&out_ptr, output_bytes));

  // execute decompression (asynchronous)
  status = nvcompCascadedDecompressAsync(
      d_comp_out,
      comp_out_bytes,
      metadata_ptr,
      metadata_bytes,
      temp_ptr,
      temp_bytes,
      out_ptr,
      output_bytes,
      stream);
  REQUIRE(status == nvcompSuccess);

  CUDA_CHECK(cudaDeviceSynchronize());

  nvcompCascadedDestroyMetadata(metadata_ptr);

  // Copy result back to host
  int res[16];
  cudaMemcpy(res, out_ptr, output_bytes, cudaMemcpyDeviceToHost);

  CUDA_CHECK(cudaFree(temp_ptr));
  CUDA_CHECK(cudaFree(d_comp_out));

  // Verify correctness
  for (size_t i = 0; i < input_size; ++i) {
    REQUIRE(res[i] == input[i]);
  }

  return 1;
}

int main(int argc, char** argv)
{
  if (argc != 1) {
    printf("ERROR: %s accepts no arguments.\n", argv[0]);
    return 1;
  }

  int num_tests = 4;
  int rv = 0;

  if (!test_rle_delta()) {
    printf("rle_delta test failed.\n");
    rv += 1;
  }

  if (!test_rle_delta_bp()) {
    printf("rle_delta_bp test failed.\n");
    rv += 1;
  }

  if (!test_ones_init_data()) {
    printf("test_ones_init_data test failed.\n");
    rv += 1;
  }

  if (!test_cascaded_backward_compatibility()) {
    printf("test_cascaded_backward_compatibility test failed.\n");
    rv += 1;
  }

  if (rv == 0) {
    printf("SUCCESS: All tests passed: %d/%d\n", (num_tests - rv), num_tests);
  } else {
    printf("FAILURE: %d/%d tests failed\n", rv, num_tests);
  }

  return rv;
}
