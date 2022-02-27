#include "CompressionPass.h"

#define ck(call) check(call, __LINE__, __FILE__)

#define CU_CHECK_SUCCESS(x)                                                         \
    do {                                                                            \
        CUresult result = x;                                                        \
        if (result != CUDA_SUCCESS)                                                 \
        {                                                                           \
            const char* msg;                                                        \
            cuGetErrorName(result, &msg);                                           \
            char errorMessage[80];                                                          \
            sprintf(errorMessage, "\n\n= CUDA Error: %d failed\n", x); \
            OutputDebugStringA(errorMessage); \
            OutputDebugStringA(msg);\
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
    outputBufferNVENC = new char[RenderConfig::getTotalSize()];

    if (mMode == Mode::Decompression) {
        initialiseDecoder();
    }
    else if (mMode == Mode::Compression) {
        initialiseEncoder();
    };
    return true;
}

bool CompressionPass::initialiseEncoder() {
    ck(CreateDXGIFactory1(__uuidof(IDXGIFactory1), (void**)pFactory.GetAddressOf()));
    ck(pFactory->EnumAdapters(iGpu, pAdapter.GetAddressOf()));

    ck(D3D12CreateDevice(pAdapter.Get(), D3D_FEATURE_LEVEL_11_0, IID_PPV_ARGS(pDevice.GetAddressOf())));

    pAdapter->GetDesc(&adapterDesc);
    char szDesc[80];
    wcstombs(szDesc, adapterDesc.Description, sizeof(szDesc));
    sprintf(msg, "GPU in use: %s\n", szDesc);
    OutputDebugStringA(msg);

    NvEncoderInitParam encodeCLIOptions("");
    enc = std::make_shared<NvEncoderD3D12>(pDevice.Get(), nWidth, nHeight, bufferFormat);

    NV_ENC_INITIALIZE_PARAMS initializeParams = { NV_ENC_INITIALIZE_PARAMS_VER };
    NV_ENC_CONFIG encodeConfig = { NV_ENC_CONFIG_VER };
    initializeParams.encodeConfig = &encodeConfig;
    enc->CreateDefaultEncoderParams(&initializeParams, encodeCLIOptions.GetEncodeGUID(), encodeCLIOptions.GetPresetGUID(), encodeCLIOptions.GetTuningInfo());
    encodeCLIOptions.SetInitParams(&initializeParams, bufferFormat);

    enc->CreateEncoder(&initializeParams);
    
    pUploadInput = std::make_unique<UploadInput>(pDevice.Get(), enc->GetNumBfrs(), enc->GetInputSize(), nWidth, nHeight, bufferFormat);

    return true;
}

bool CompressionPass::initialiseDecoder2() {
    /*NVDEC*/
    // query capabilities of GPU
    CUVIDDECODECAPS decodeCaps = {};
    CUresult result;
    int nGpu;
    char szDeviceName[40];

    CU_CHECK_SUCCESS(cuInit(0));
    cuDeviceGetCount(&nGpu);

    CU_CHECK_SUCCESS(cuDeviceGet(&cuDevice, 0));
    CU_CHECK_SUCCESS(cuDeviceGetName(szDeviceName, sizeof(szDeviceName), cuDevice));
    sprintf(msg, "GPU in use: %s\n", szDeviceName);
    OutputDebugStringA(msg);

    CU_CHECK_SUCCESS(cuCtxCreate(&cuContext, 0, cuDevice));

    CU_CHECK_SUCCESS(cuDevicePrimaryCtxRetain(&cuContext, cuDevice));


    // set IN params for decodeCaps
    decodeCaps.eCodecType = cudaVideoCodec_H264;//H264
    decodeCaps.eChromaFormat = cudaVideoChromaFormat_420;//YUV 4:4:4
    decodeCaps.nBitDepthMinus8 = 0;// 8 bit
    result = cuvidGetDecoderCaps(&decodeCaps);

    // Check if content is supported
    if (!decodeCaps.bIsSupported) {
        NVDEC_THROW_ERROR("Codec not supported on this GPU", CUDA_ERROR_NOT_SUPPORTED);
    }

    unsigned int coded_width = 1920;
    unsigned int coded_height = 1080;
    // validate the content resolution supported on underlying hardware
    if ((coded_width > decodeCaps.nMaxWidth) ||
        (coded_height > decodeCaps.nMaxHeight)) {
        NVDEC_THROW_ERROR("Resolution not supported on this GPU",
            CUDA_ERROR_NOT_SUPPORTED);
    }
    // Max supported macroblock count CodedWidth*CodedHeight/256 must be <= nMaxMBCount
    if ((coded_width >> 4) * (coded_height >> 4) > decodeCaps.nMaxMBCount) {
        NVDEC_THROW_ERROR("MBCount not supported on this GPU",
            CUDA_ERROR_NOT_SUPPORTED);
    }

    CUVIDDECODECREATEINFO DecodeCreateInfo = { 0 };
    DecodeCreateInfo.CodecType = cudaVideoCodec_H264;
    DecodeCreateInfo.ChromaFormat = cudaVideoChromaFormat_420;
    DecodeCreateInfo.OutputFormat = cudaVideoSurfaceFormat_NV12;
    DecodeCreateInfo.bitDepthMinus8 = 0;

    DecodeCreateInfo.ulNumOutputSurfaces = 2;
    DecodeCreateInfo.ulNumDecodeSurfaces = 2;
    DecodeCreateInfo.ulWidth = 1920;       // not sure
    DecodeCreateInfo.ulHeight = 1080;      // not sure

    DecodeCreateInfo.ulMaxWidth = 1920;
    DecodeCreateInfo.ulMaxHeight = 1080;

    DecodeCreateInfo.ulTargetWidth = DecodeCreateInfo.ulWidth;
    DecodeCreateInfo.ulTargetHeight = DecodeCreateInfo.ulHeight;

    CU_CHECK_SUCCESS(cuvidCreateDecoder(&m_hDecoder, &DecodeCreateInfo));
    OutputDebugString(L"Finished creating decoder...\n");
    
    // havent found a way to return false when there is error
    return true;
}

bool CompressionPass::initialiseDecoder() {
    Rect cropRect = {0,0,0,0};
    Dim resizeDim = {0,0};
    unsigned int opPoint = 0;
    bool bDispAllLayers = false;

    ck(cuInit(0));
    //below is the code for createCudaContext(&cuContext, iGpu, 0);
    ck(cuDeviceGet(&cuDevice, iGpu));
    char szDeviceName[80];
    ck(cuDeviceGetName(szDeviceName, sizeof(szDeviceName), cuDevice));
    int flags = 0;
    cuCtxCreate(&cuContext, flags, cuDevice);

    dec = std::make_shared<NvDecoder>(cuContext, false, cudaVideoCodec_H264, false, false, &cropRect, &resizeDim);

    /* Set operating point for AV1 SVC. It has no impact for other profiles or codecs
     * PFNVIDOPPOINTCALLBACK Callback from video parser will pick operating point set to NvDecoder  */
    dec->SetOperatingPoint(opPoint, bDispAllLayers);

    return 0;
}

class MyDataProvider : FFmpegDemuxer::DataProvider {
public:
    uint8_t* pBuf; 
    int nBuf;
    
    MyDataProvider(uint8_t* pBuf, int nBuf) {
        pBuf = pBuf;
        nBuf = nBuf;
    }
    int GetData(uint8_t* pBuf, int nBuf) {
        
    }
};

void CompressionPass::DecodeMediaFile()
{
    if (demuxer == nullptr) {
        demuxer = new FFmpegDemuxer((char*)RenderConfig::mConfig[0].compressionPassOutputLocation2);
    }
    std::lock_guard lock(NetworkManager::mMutexClientVisTexRead);
    demuxer->Demux(&pVideo, &nVideoBytes);
    sprintf(msg, "\nDecodeMediaFile: original size: %d, demux size: %d\n", RenderConfig::mConfig[0].compressedSize2, nVideoBytes);
    OutputDebugStringA(msg);
    nFrameReturned = dec->Decode(pVideo, nVideoBytes);
    if (!nFrame && nFrameReturned)
        LOG(INFO) << dec->GetVideoInfo();

    for (int i = 0; i < nFrameReturned; i++) {
        pFrame = dec->GetFrame();
        // dump YUV to disk
        if (dec->GetWidth() == dec->GetDecodeWidth())
        {
            memcpy(outputBufferNVENC, pFrame, dec->GetFrameSize());
        }
        else
        {
            OutputDebugString(L"\nPANIC I DONT KNOW WHAT TO DO\n");
        }
        std::vector <std::string> aszDecodeOutFormat = { "NV12", "P016", "YUV444", "YUV444P16" };
        sprintf(msg, "\nSize of frame: %d\nTotal frame decoded: %d\nSaved in format: %s", dec->GetFrameSize(),
            nFrame, aszDecodeOutFormat[dec->GetOutputFormat()].c_str());
        OutputDebugStringA(msg);
    }
    nFrame += nFrameReturned;
}

void CompressionPass::initScene(RenderContext* pRenderContext, Scene::SharedPtr pScene)
{
}

void CompressionPass::execute(RenderContext* pRenderContext)
{
    bool LZ4 = true;
    if (LZ4) {
        executeLZ4(pRenderContext);
    }
    executeNVENC(pRenderContext);
}

void CompressionPass::executeNVENC(RenderContext* pRenderContext)
{
    if (mMode == Mode::Compression) {
        std::lock_guard lock(NetworkManager::mMutexServerVisTexRead);

        std::vector<std::vector<uint8_t>> vPacket;

        // Get next available input buffer. CPU wait in NvEncoderD3D12::GetEncodedPacket() 
        // ensures that NVENC has finished processing on this input buffer
        const NvEncInputFrame* encoderInputFrame = enc->GetNextInputFrame();

        NV_ENC_FENCE_POINT_D3D12* pInpFencePOint = enc->GetInpFencePoint();

        //hard coded i=0 at the moment
        char* sourceBuffer = (char*)RenderConfig::mConfig[0].networkPassOutputLocation;
        int sourceBufferSize = RenderConfig::BufferTypeToSize(RenderConfig::mConfig[0].type);
        pUploadInput->ReadInputFrame(sourceBuffer, encoderInputFrame, pInpFencePOint);

        enc->EncodeFrame(vPacket);

        nFrame += (int)vPacket.size();
        size_t compressedSize = 0;
        for (std::vector<uint8_t>& packet : vPacket)
        {
            compressedSize += packet.size();
            memcpy(&outputBufferNVENC[RenderConfig::mConfig[0].compressedSize2 + compressedSize], packet.data(), packet.size());
            //fpOut.write(reinterpret_cast<char*>(packet.data()), packet.size());
        }
        sprintf(msg, "\nSize of NVENC compressed output: %zu\n", compressedSize);
        RenderConfig::mConfig[0].compressionPassOutputLocation2 = outputBufferNVENC;
        RenderConfig::mConfig[0].compressedSize2 += (int)compressedSize;
        OutputDebugStringA(msg);
    }
    else if (mMode == Mode::Decompression) {
        DecodeMediaFile();
    }
}

void CompressionPass::executeLZ4(RenderContext* pRenderContext)
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
                const char* const sourceBuffer = reinterpret_cast<const char* const>(RenderConfig::mConfig[i].compressionPassOutputLocation);
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
