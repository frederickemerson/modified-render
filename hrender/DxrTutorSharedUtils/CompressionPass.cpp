#include "CompressionPass.h"

bool CompressionPass::initialize(RenderContext* pRenderContext, ResourceManager::SharedPtr pResManager)
{
    int sizeToAllocateOutputBuffer = RenderConfig::getTotalSize();
    outputBuffer = new char[RenderConfig::getTotalSize()];

    return true;
}

void CompressionPass::initScene(RenderContext* pRenderContext, Scene::SharedPtr pScene)
{
}

void CompressionPass::execute(RenderContext* pRenderContext)
{
    // ===================== COMPRESSION ===================== //
    if (mMode == Mode::Compression) {
        // Loop over all textures, compress each one
        for (int i = 0; i < RenderConfig::mConfig.size(); i++) {
            {
                std::lock_guard lock(NetworkManager::mMutexServerVisTexRead);

                // Parameters for Compression
                const char* const sourceBuffer = reinterpret_cast<const char* const>(RenderConfig::mConfig[i].cpuLocation);
                int sourceBufferSize = RenderConfig::BufferTypeToSize(RenderConfig::mConfig[i].type);

                // Compress buffer
                int compressedSize = LZ4_compress_default(sourceBuffer, outputBuffer, sourceBufferSize, sourceBufferSize);

                // Update size of compressed buffer
                RenderConfig::mConfig[i].compressedSize = compressedSize;

                // Update location of buffer
                RenderConfig::mConfig[i].cpuLocation = outputBuffer;

                char buffer[140];
                sprintf(buffer, "\n\n= Compressed Buffer: Original size: %d, Compressed size: %d =========", sourceBufferSize, compressedSize);
                OutputDebugStringA(buffer);
            }
        }
    }


    // ===================== DECOMPRESSION ===================== //
    else { // mMode == Mode::Decompression
        // Loop over all buffers, decompress each one
        for (int i = 0; i < RenderConfig::mConfig.size(); i++) {
            {
                std::lock_guard lock(NetworkManager::mMutexClientVisTexRead);
                // Parameters for Decompression
                const char* const sourceBuffer = reinterpret_cast<const char* const>(RenderConfig::mConfig[i].cpuLocation);
                int sourceBufferSize = RenderConfig::mConfig[i].compressedSize;
                int maxDecompressedSize = RenderConfig::BufferTypeToSize(RenderConfig::mConfig[i].type);

                // Decompress buffer
                int decompressedSize = LZ4_decompress_safe(sourceBuffer, outputBuffer, sourceBufferSize, maxDecompressedSize);

                // Update size of decompressed buffer
                RenderConfig::mConfig[i].compressedSize = decompressedSize;

                // Update location of buffer
                RenderConfig::mConfig[i].cpuLocation = outputBuffer;
            }
        }
    }
}

void CompressionPass::renderGui(Gui::Window* pPassWindow)
{
    int dirty = 0;
    pPassWindow->text(mMode == Mode::Compression ? "Compression"
        : mMode == Mode::Decompression? "Decompression"
        : "Unknown Compression Pass");

    // If any of our UI parameters changed, let the pipeline know we're doing something different next frame
    if (dirty) setRefreshFlag();
}
