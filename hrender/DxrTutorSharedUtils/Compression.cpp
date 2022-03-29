#include "Compression.h"

int Compression::executeLZ4Compress(const char* sourceBuffer, char* destinationBuffer, int sourceBufferSize)
{
    // ===================== COMPRESSION ===================== //
        int compressedSize = LZ4_compress_default(sourceBuffer, destinationBuffer, sourceBufferSize, sourceBufferSize);
        char buffer[140];
        sprintf(buffer, "\n\n= Compressed Buffer: Original size: %d, Compressed size: %d =========", sourceBufferSize, compressedSize);
        OutputDebugStringA(buffer);
        return compressedSize;
}

int Compression::executeLZ4Decompress(const char* sourceBuffer, char* destinationBuffer, int sourceBufferSize, int maxDecompressedSize) {
    // ===================== DECOMPRESSION ===================== //
    int decompressedSize = LZ4_decompress_safe(sourceBuffer, destinationBuffer, sourceBufferSize, maxDecompressedSize);
    char buffer[140];
    sprintf(buffer, "\n\n= Compressed Buffer: Original size: %d, Decompressed size: %d =========", sourceBufferSize, decompressedSize);
    OutputDebugStringA(buffer);
    return decompressedSize;
}