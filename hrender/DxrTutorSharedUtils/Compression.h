#pragma once

#include "lz4.h"
#include <memory>

class Compression {

public:
    static int executeLZ4Compress(const char* sourceBuffer, char* destinationBuffer, int sourceBufferSize);
    static int executeLZ4Decompress(const char* sourceBuffer, char* destinationBuffer, int sourceBufferSize, int maxDecompressedSize);
};