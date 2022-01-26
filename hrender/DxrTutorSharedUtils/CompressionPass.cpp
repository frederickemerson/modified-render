#include "CompressionPass.h"
//#include "cuda_runtime_api.h"

#define CU_CHECK_SUCCESS(x)                                                         \
    do {                                                                            \
        CUresult result = x;                                                        \
        if (result != CUDA_SUCCESS)                                                 \
        {                                                                           \
            const char* msg;                                                        \
            cuGetErrorName(result, &msg);                                           \
            char errorMessage[80];                                                          \
            sprintf(errorMessage, "\n\n= CUDA Error: %d failed\n", x); \
            OutputDebugStringA(errorMessage);\
            return 0;                                                               \
        }                                                                           \
    } while(0)

#define CUDA_CHECK_SUCCESS(x)                                                                            \
    do {                                                                                                 \
        cudaError_t result = x;                                                                          \
        if (result != cudaSuccess)                                                                       \
        {                                                                                                \
            logError("CUDA Error: " #x " failed with error " + std::string(cudaGetErrorString(result))); \
            return 0;                                                                                    \
        }                                                                                                \
    } while(0)

bool CompressionPass::initialize(RenderContext* pRenderContext, ResourceManager::SharedPtr pResManager)
{
    int sizeToAllocateOutputBuffer = RenderConfig::getTotalSize();
    outputBuffer = new char[RenderConfig::getTotalSize()];

    ///*NVDEC*/
    //// query capabilities of GPU
    //CUVIDDECODECAPS decodeCaps = {};
    //CUresult result;

    //cuInit(0);
    //OutputDebugString(L"try device get");
    //CU_CHECK_SUCCESS(cuDeviceGet(cuDevice, 0));
    //OutputDebugString(L"device get");
    //cuDevicePrimaryCtxRetain(cuContext, *cuDevice);


    //// set IN params for decodeCaps
    //decodeCaps.eCodecType = cudaVideoCodec_H264;//H264
    //decodeCaps.eChromaFormat = cudaVideoChromaFormat_444;//YUV 4:4:4
    //decodeCaps.nBitDepthMinus8 = 0;// 8 bit
    //result = cuvidGetDecoderCaps(&decodeCaps);

    //// Check if content is supported
    //if (!decodeCaps.bIsSupported) {
    //    NVDEC_THROW_ERROR("Codec not supported on this GPU", CUDA_ERROR_NOT_SUPPORTED);
    //}

    //unsigned int coded_width = 1920;
    //unsigned int coded_height = 1080;
    //// validate the content resolution supported on underlying hardware
    //if ((coded_width > decodeCaps.nMaxWidth) ||
    //    (coded_height > decodeCaps.nMaxHeight)) {
    //    NVDEC_THROW_ERROR("Resolution not supported on this GPU",
    //        CUDA_ERROR_NOT_SUPPORTED);
    //}
    //// Max supported macroblock count CodedWidth*CodedHeight/256 must be <= nMaxMBCount
    //if ((coded_width >> 4) * (coded_height >> 4) > decodeCaps.nMaxMBCount) {
    //    NVDEC_THROW_ERROR("MBCount not supported on this GPU",
    //        CUDA_ERROR_NOT_SUPPORTED);
    //}
    //
    //CUVIDDECODECREATEINFO DecodeCreateInfo = { 0 };
    //DecodeCreateInfo.CodecType = cudaVideoCodec_H264;
    //DecodeCreateInfo.ChromaFormat = cudaVideoChromaFormat_444;
    //DecodeCreateInfo.OutputFormat = cudaVideoSurfaceFormat_YUV444;
    //DecodeCreateInfo.bitDepthMinus8 = 0;

    //DecodeCreateInfo.ulNumOutputSurfaces = 2;
    //DecodeCreateInfo.ulNumDecodeSurfaces = 2;
    //DecodeCreateInfo.ulWidth = 1920;       // not sure
    //DecodeCreateInfo.ulHeight = 1080;      // not sure

    //DecodeCreateInfo.ulMaxWidth = 1920;
    //DecodeCreateInfo.ulMaxHeight = 1080;

    //DecodeCreateInfo.ulTargetWidth = DecodeCreateInfo.ulWidth;
    //DecodeCreateInfo.ulTargetHeight = DecodeCreateInfo.ulHeight;

    ////OutputDebugString(L"Video Decoding Params:" + "\nNum Surfaces : " + DecodeCreateInfo.ulNumDecodeSurfaces
    ////    + "\nCrop         : [" << DecodeCreateInfo.display_area.left + ", " + DecodeCreateInfo.display_area.top + ", "
    ////    + DecodeCreateInfo.display_area.right + ", " + DecodeCreateInfo.display_area.bottom << "]" + std::endl
    ////    "\nResize       : " << DecodeCreateInfo.ulTargetWidth << "x" << DecodeCreateInfo.ulTargetHeight << std::endl
    ////    "\nDeinterlace  : " << std::vector<const char*>{"Weave", "Bob", "Adaptive"} [DecodeCreateInfo.DeinterlaceMode]);
    ////m_videoInfo << std::endl;
    //cuvidCreateDecoder(&m_hDecoder, &DecodeCreateInfo);
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
                const char* const sourceBuffer = reinterpret_cast<const char* const>(RenderConfig::mConfig[i].networkPassOutputLocation);
                int sourceBufferSize = RenderConfig::BufferTypeToSize(RenderConfig::mConfig[i].type);

                // Compress buffer
                int compressedSize = LZ4_compress_default(sourceBuffer, outputBuffer, sourceBufferSize, sourceBufferSize);
                if (compressedSize == 0) {
                    OutputDebugString(L"\nError: Compression failed\n");
                }

                // Update size of compressed buffer
                RenderConfig::mConfig[i].compressedSize = compressedSize;

                // Update location of buffer
                RenderConfig::mConfig[i].compressionPassOutputLocation = outputBuffer;

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
                const char* const sourceBuffer = reinterpret_cast<const char* const>(RenderConfig::mConfig[i].networkPassOutputLocation);
                int sourceBufferSize = RenderConfig::mConfig[i].compressedSize;
                int maxDecompressedSize = RenderConfig::BufferTypeToSize(RenderConfig::mConfig[i].type);

                if (sourceBufferSize == maxDecompressedSize) {
                    OutputDebugString(L"Skipping decompression, texture didnt change");
                    return;
                }

                // Decompress buffer
                int decompressedSize = LZ4_decompress_safe(sourceBuffer, outputBuffer, sourceBufferSize, maxDecompressedSize);
                if (decompressedSize <= 0) {
                    OutputDebugString(L"\nError: Decompression failed\n");
                }

                char buffer[140];
                sprintf(buffer, "\n\n= Compressed Buffer: Original size: %d, Decompressed size: %d =========", sourceBufferSize, decompressedSize);
                OutputDebugStringA(buffer);

                // Update size of decompressed buffer
                RenderConfig::mConfig[i].compressedSize = decompressedSize;

                // Update location of buffer
                RenderConfig::mConfig[i].compressionPassOutputLocation = outputBuffer;
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
