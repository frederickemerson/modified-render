#include "compression.h"
#include "nvcomp/cascaded.hpp"
#include "nvcomp.hpp"


// return pointer to data and set uncompressed_bytes to output byte size
void* PlsWork::compress(size_t* bytes, std::vector<uint8_t> uncompressed_data)
{
    size_t uncompressed_bytes = *bytes;
    size_t temp_bytes;
    size_t output_bytes;

    nvcomp::CascadedCompressor compressor(nvcomp::TypeOf<int>());

    compressor.configure(uncompressed_bytes, &temp_bytes, &output_bytes);

    void* temp_space;
    cudaMalloc(&temp_space, temp_bytes);

    void* output_space;
    cudaMalloc(&output_space, output_bytes);

    //size_t* d_compressed_bytes;
    //cudaMallocHost(&d_compressed_bytes, sizeof(size_t));

    // Create CUDA stream
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    compressor.compress_async(&uncompressed_data, uncompressed_bytes,
        temp_space, temp_bytes, output_space, bytes, stream);

    cudaStreamSynchronize(stream);
    
    return output_space;
}
