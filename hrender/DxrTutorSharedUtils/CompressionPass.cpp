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
    int sizeToAllocateOutputBuffer = VIS_TEX_LEN;
    outputBuffer = new char[sizeToAllocateOutputBuffer];
    outputBufferNVENC = new char[sizeToAllocateOutputBuffer];
    
    mDisplayableBuffers.push_back({ 0, "LZ4" });
    mDisplayableBuffers.push_back({ 1, "H264" });

    mNumOfTextures = mHybridMode ? 1 : 1;

    if (mMode == Mode::Compression) {
        if (mHybridMode) {
            initialiseH264Encoders();
        }
        else {
            initialiseH264RemoteEncoder();
        }
        //initialiseEncoder();
    } else if (mMode == Mode::Decompression) {
        if (mHybridMode) {
            initialiseH264Decoders();
            // 24 out of 32 bits are unused for AO, so we just keep them as 0.
            //uint8_t* pOutputBuffer = (uint8_t*)outputBuffer + VIS_TEX_LEN;
            //for (int i = 0; i < VIS_TEX_LEN; i++) {
            //    if (i % 4 == 0) continue;
            //    pOutputBuffer[i] = 0;
            //}

        }
        else {
            initialiseH264RemoteDecoder();
        }
        //initialiseDecoder();
    };
    
    return true;
}

bool CompressionPass::initialiseH264HybridEncoder(CodecParams* codecParams) {
    AVCodec* h264enc;
    if (isUsingCPU) {
        h264enc = avcodec_find_encoder(codecParams->codecID);
    }
    else {
        h264enc = avcodec_find_encoder_by_name(codecParams->codecName);
    }

    mpCodecContext = avcodec_alloc_context3(h264enc);
    if (!mpCodecContext) {
        OutputDebugString(L"\nError: Could not allocated codec context.\n");
        return false;
    }

    mpCodecContexts.push_back(mpCodecContext);

    /* Set up parameters for the context */
    mpCodecContext->width = codecParams->width;
    mpCodecContext->height = codecParams->height;
    mpCodecContext->pix_fmt = codecParams->outPixFmt;

    mpCodecContext->time_base = av_make_q(1, 90000);
    int64_t bitrate = 2000000;
    mpCodecContext->bit_rate = bitrate;
    mpCodecContext->framerate = AVRational{ 1, 60 };
    mpCodecContext->ticks_per_frame = 2;
    mpCodecContext->thread_count = 0; // 0 makes FFmpeg choose the optimal number
    mpCodecContext->thread_type = FF_THREAD_SLICE;

    AVDictionary* param = nullptr;
    av_dict_set(&param, "crf", "0", 0);
    if (isUsingCPU) {
        av_dict_set(&param, "profile:v", "high", 0);
        av_dict_set(&param, "preset", "slower", 0);
        av_dict_set(&param, "tune", "zerolatency", 0);
        //av_dict_set(&param, "-b:v", "3000k", 0);
        av_dict_set(&param, "refs", "1", 0);
        av_dict_set(&param, "me_range", "16", 0);
        //av_dict_set(&param, "intra-refresh", "1", 0);
        av_dict_set(&param, "g", "1", 0);
        //av_dict_set(&param, "vsync", "0", 0);
    }
    else {
        av_dict_set(&param, "preset", "llhq", 0);
        av_dict_set(&param, "profile:v", "high", 0);
        //av_dict_set(&param, "rc", "cbr_ld_hq", 0);
        //av_dict_set(&param, "vsync", "0", 0);
        av_dict_set(&param, "g", "1", 0);
    }

    /* Open the context for encoding */
    int err = avcodec_open2(mpCodecContext, h264enc, &param);
    if (err < 0) {
        av_strerror(err, msg, 100);
        OutputDebugString(L"\nFailure to open encoder context due to: ");
        OutputDebugStringA(msg);
    }

    if (codecParams->isColorConvertNeeded) {
        /* Initialise the color converter */
        mpSwsContexts.push_back(sws_getContext(
            codecParams->width, codecParams->height, codecParams->inPixFmt,
            codecParams->width, codecParams->height, codecParams->outPixFmt,
            SWS_POINT, nullptr, nullptr, nullptr));
    }
    else {
        mpSwsContexts.push_back(nullptr);
    }

    /* Initialise the AVFrame and AVPacket*/
    mpFrame = av_frame_alloc();
    mpPacket = av_packet_alloc();

    return true;
}

bool CompressionPass::initialiseH264HybridDecoder(CodecParams* codecParams) {
    AVCodec* h264dec = avcodec_find_decoder(codecParams->codecID);
    mpCodecContext = avcodec_alloc_context3(h264dec);
    if (!mpCodecContext) {
        OutputDebugString(L"\nError: Could not allocated codec context.\n");
        return false;
    }

    mpCodecContexts.push_back(mpCodecContext);

    /* Set params for the context */
    mpCodecContext->width = codecParams->width;
    mpCodecContext->height = codecParams->height;
    mpCodecContext->pix_fmt = codecParams->inPixFmt;

    mpCodecContext->ticks_per_frame = 2;
    mpCodecContext->thread_count = 0; // 0 makes FFmpeg choose the optimal number
    mpCodecContext->thread_type = FF_THREAD_SLICE;

    /* Open the context for decoding */
    int err = avcodec_open2(mpCodecContext, h264dec, NULL);
    if (err < 0) {
        av_strerror(err, msg, 120);
        OutputDebugString(L"\nFailure to open decoder context due to: ");
        OutputDebugStringA(msg);
    }

    if (codecParams->isColorConvertNeeded) {
        /* Initialise the color converter*/
        mpSwsContexts.push_back(sws_getContext(
            codecParams->width, codecParams->height, codecParams->inPixFmt,
            codecParams->width, codecParams->height, codecParams->outPixFmt,
            SWS_BILINEAR, nullptr, nullptr, nullptr
        ));
    }
    else {
        mpSwsContexts.push_back(nullptr);
    }
    /* Initialise the AVFrame and AVPacket*/
    mpFrame = av_frame_alloc();
    mpPacket = av_packet_alloc();

    return true;
}

// Remote encoder/decoder is initialised separately out of convenience
bool CompressionPass::initialiseH264RemoteEncoder() {
    AVCodec* h264enc;
    if (isUsingCPU) {
        h264enc = avcodec_find_encoder(AV_CODEC_ID_H264);
    }
    else {
        h264enc = avcodec_find_encoder_by_name("h264_nvenc");
    }

    mpCodecContext = avcodec_alloc_context3(h264enc);
    if (!mpCodecContext) OutputDebugString(L"\nError: Could not allocated codec context.\n");

    mpCodecContexts.push_back(mpCodecContext);

    /* Set up parameters for the context */
    mpCodecContext->width = nWidth;
    mpCodecContext->height = nHeight;
    mpCodecContext->time_base = av_make_q(1, 90000);
    int64_t bitrate = 2000000;
    mpCodecContext->bit_rate = bitrate;
    mpCodecContext->framerate = AVRational{ 1, 60 };
    mpCodecContext->ticks_per_frame = 2;
    mpCodecContext->thread_count = 0; // 0 makes FFmpeg choose the optimal number
    mpCodecContext->thread_type = FF_THREAD_SLICE;
    mpCodecContext->pix_fmt = AV_PIX_FMT_YUV444P;

    AVDictionary* param = nullptr;
    av_dict_set(&param, "crf", "0", 0);
    if (isUsingCPU) {
        av_dict_set(&param, "profile:v", "high", 0);
        av_dict_set(&param, "preset", "slower", 0);
        av_dict_set(&param, "tune", "zerolatency", 0);
        //av_dict_set(&param, "-b:v", "3000k", 0);
        av_dict_set(&param, "refs", "1", 0);
        av_dict_set(&param, "me_range", "16", 0);
        //av_dict_set(&param, "intra-refresh", "1", 0);
        av_dict_set(&param, "g", "1", 0);
        //av_dict_set(&param, "vsync", "0", 0);
    }
    else {
        av_dict_set(&param, "preset", "llhq", 0);
        av_dict_set(&param, "profile:v", "high", 0);
        //av_dict_set(&param, "rc", "cbr_ld_hq", 0);
        //av_dict_set(&param, "vsync", "0", 0);
        av_dict_set(&param, "g", "1", 0);
    }


    /* Open the context for encoding */
    int err = avcodec_open2(mpCodecContext, h264enc, &param);
    if (err < 0) {
        av_strerror(err, msg, 100);
        OutputDebugString(L"\nFailure to open encoder context due to: ");
        OutputDebugStringA(msg);
    }

    mpSwsContexts.push_back(nullptr);

    /* Initialise the AVFrame and AVPacket*/
    mpFrame = av_frame_alloc();
    mpPacket = av_packet_alloc();

    return true;
}

bool CompressionPass::initialiseH264RemoteDecoder() {
    AVCodec* h264dec = avcodec_find_decoder(AV_CODEC_ID_H264);
    mpCodecContext = avcodec_alloc_context3(h264dec);
    if (!mpCodecContext) OutputDebugString(L"\nError: Could not allocated codec context.\n");

    mpCodecContexts.push_back(mpCodecContext);

    /* Set params for the context */
    mpCodecContext->width = nWidth;
    mpCodecContext->height = nHeight;
    mpCodecContext->ticks_per_frame = 2;
    mpCodecContext->thread_count = 0; // 0 makes FFmpeg choose the optimal number
    mpCodecContext->thread_type = FF_THREAD_SLICE;
    mpCodecContext->pix_fmt = AV_PIX_FMT_YUV444P;

    AVDictionary* param = nullptr;
    av_dict_set(&param, "qp", "0", 0);
    av_dict_set(&param, "preset", "ultrafast", 0);
    av_dict_set(&param, "tune", "zerolatency", 0);

    /* Open the context for decoding */
    int err = avcodec_open2(mpCodecContext, h264dec, NULL);
    if (err < 0) {
        av_strerror(err, msg, 120);
        OutputDebugString(L"\nFailure to open decoder context due to: ");
        OutputDebugStringA(msg);
    }

    mpSwsContexts.push_back(nullptr);

    /* Initialise the AVFrame and AVPacket*/
    mpFrame = av_frame_alloc();
    mpPacket = av_packet_alloc();
    return true;
}

bool CompressionPass::initialiseH264Encoders() {
    // First encoder: VisTex Encoder
    CodecParams params = CodecParams();
    params.codecID = AV_CODEC_ID_H264;
    params.width = nWidth;
    params.height = nHeight;
    params.isColorConvertNeeded = true;
    params.inPixFmt = AV_PIX_FMT_RGBA;
    params.outPixFmt = AV_PIX_FMT_YUV444P;

    initialiseH264HybridEncoder(&params);

    // Second encoder: AO Encoder
    params = CodecParams();
    params.codecID = AV_CODEC_ID_H264;
    params.width = nWidth;
    params.height = nHeight;
    params.isColorConvertNeeded = false;
    params.inPixFmt = AV_PIX_FMT_GRAY8;
    params.outPixFmt = AV_PIX_FMT_GRAY8;

    initialiseH264HybridEncoder(&params);

    return true;
}

bool CompressionPass::initialiseH264Decoders() {
    // First decoder: VisTex Decoder
    CodecParams params = CodecParams();
    params.codecID = AV_CODEC_ID_H264;
    params.width = nWidth;
    params.height = nHeight;
    params.isColorConvertNeeded = true;
    params.inPixFmt = AV_PIX_FMT_YUV444P;
    params.outPixFmt = AV_PIX_FMT_RGBA;

    initialiseH264HybridDecoder(&params);

    // Second decoder: AO Decoder
    params = CodecParams();
    params.codecID = AV_CODEC_ID_H264;
    params.width = nWidth;
    params.height = nHeight;
    params.isColorConvertNeeded = false;
    params.inPixFmt = AV_PIX_FMT_GRAY8;
    params.outPixFmt = AV_PIX_FMT_GRAY8;

    initialiseH264HybridDecoder(&params);

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
    
    MyDataProvider(uint8_t* pbuf, int nbuf) {
        pBuf = pbuf;
        nBuf = nbuf;
    }
    int GetData(uint8_t* pBuf, int nBuf) {
        
    }
};

void CompressionPass::DecodeMediaFile()
{
    if (demuxer == nullptr) {
        demuxer = new FFmpegDemuxer(mGetInputBuffer());
    }
    std::lock_guard lock(ClientNetworkManager::mMutexClientVisTexRead);
    demuxer->Demux(&pVideo, &nVideoBytes);
    sprintf(msg, "\nDecodeMediaFile: original size: %d, demux size: %d\n", RenderConfig::mConfig[0].compressedSize2, nVideoBytes);
    OutputDebugStringA(msg);
    nFrameReturned = dec->Decode((uint8_t*)mGetInputBuffer(), RenderConfig::mConfig[0].compressedSize2 * 2);
    if (!nFrame && nFrameReturned)
        LOG(INFO) << dec->GetVideoInfo();

    //for (int i = 0; i < nFrameReturned; i++) {
    if (nFrameReturned) {
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
    nFrame += 1; //nFrameReturned;
}

void CompressionPass::initScene(RenderContext* pRenderContext, Scene::SharedPtr pScene)
{
}

void CompressionPass::execute(RenderContext* pRenderContext)
{
    if (mCodecType == LZ4) {
        executeLZ4(pRenderContext);
    }
    else if (mCodecType == H264) {
        executeH264(pRenderContext);
       /* if (mMode == Mode::Compression) {
            executeNVENC(pRenderContext);
        }
        else {
            executeH264(pRenderContext);
        }*/
    }
    
}

void CompressionPass::executeNVENC(RenderContext* pRenderContext)
{
    if (mMode == Mode::Compression) {
        std::lock_guard lock(ServerNetworkManager::mMutexServerVisTexRead);

        std::vector<std::vector<uint8_t>> vPacket;

        // Get next available input buffer. CPU wait in NvEncoderD3D12::GetEncodedPacket() 
        // ensures that NVENC has finished processing on this input buffer
        const NvEncInputFrame* encoderInputFrame = enc->GetNextInputFrame();

        NV_ENC_FENCE_POINT_D3D12* pInpFencePOint = enc->GetInpFencePoint();

        //hard coded i=0 at the moment
        int sourceBufferSize = RenderConfig::BufferTypeToSize(RenderConfig::mConfig[0].type);
        pUploadInput->ReadInputFrame(mGetInputBuffer(), encoderInputFrame, pInpFencePOint);

        enc->EncodeFrame(vPacket);

        nFrame += (int)vPacket.size();
        size_t compressedSize = 0;
        for (std::vector<uint8_t>& packet : vPacket)
        {
            compressedSize += packet.size();
            uint8_t* data = packet.data();
            memcpy(&outputBufferNVENC[compressedSize], packet.data(), packet.size());
            //fpOut.write(reinterpret_cast<char*>(packet.data()), packet.size());
        }
        sprintf(msg, "\nSize of NVENC compressed output: %zu\n", compressedSize);
        //RenderConfig::mConfig[0].compressionPassOutputLocation2 = outputBufferNVENC;
        RenderConfig::mConfig[0].compressedSize2 = (int)compressedSize;
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
                std::lock_guard lock(ServerNetworkManager::mMutexServerVisTexRead);

                // Parameters for Compression
                const char* const sourceBuffer = reinterpret_cast<const char* const>(mGetInputBuffer());
                int sourceBufferSize = VIS_TEX_LEN;

                // Compress buffer
                int compressedSize = LZ4_compress_default(sourceBuffer, outputBuffer, sourceBufferSize, sourceBufferSize);
                if (compressedSize == 0) {
                    OutputDebugString(L"\nError: Compression failed\n");
                }

                // Update size of compressed buffer
                outputBufferSize = compressedSize;

                // Update location of buffer
                //RenderConfig::mConfig[i].compressionPassOutputLocation = outputBuffer;

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
                std::lock_guard lock(ClientNetworkManager::mMutexClientVisTexRead);
                // Parameters for Decompression
                const char* const sourceBuffer = reinterpret_cast<const char* const>(mGetInputBuffer());
                int sourceBufferSize = mGetInputBufferSize();
                int maxDecompressedSize = VIS_TEX_LEN;

                if (sourceBufferSize == maxDecompressedSize) {
                    OutputDebugString(L"Skipping decompression, texture didnt change");
                    return;
                }

                // Decompress buffer
                int decompressedSize = LZ4_decompress_safe(sourceBuffer, (char*)outputBuffer, sourceBufferSize, maxDecompressedSize);
                if (decompressedSize <= 0) {
                    OutputDebugString(L"\nError: Decompression failed\n");
                }

                char buffer[140];
                sprintf(buffer, "\n\n= Compressed Buffer: Original size: %d, Decompressed size: %d =========", sourceBufferSize, decompressedSize);
                OutputDebugStringA(buffer);

                // Update size of decompressed buffer
                outputBufferSize = decompressedSize;

            }
        }
    }
}

void CompressionPass::executeH264(RenderContext* pRenderContext)
{
    /* H264 compression using FFmpeg API*/

    // ===================== COMPRESSION ===================== //
    if (mMode == Mode::Compression) {
        char* pOutputBuffer = outputBuffer;
        std::vector<int> compressedSizes; // Stores compressed sizes of all buffers
        outputBufferSize = 0;

        // Loop over all textures, compress each one
        for (int i = 0; i < mNumOfTextures; i++)
        {
            std::lock_guard lock(ServerNetworkManager::mMutexServerVisTexRead);
 
            // Parameters for Compression
            uint8_t* sourceBuffer = &reinterpret_cast<uint8_t*>(mGetInputBuffer())[mBufferOffsets[i]];
            int sourceBufferSize = mBufferSizes[i];
            mpCodecContext = mpCodecContexts[i];
            mpSwsContext = mpSwsContexts[i];

            /* Set up AVFrame/AVPacket params and buffers */
            mpFrame->format = mpCodecContext->pix_fmt;
            mpFrame->width = mpCodecContext->width;
            mpFrame->height = mpCodecContext->height;
            mpFrame->pts = 0;

            if (av_frame_get_buffer(mpFrame, 0) < 0)
            {
                OutputDebugString(L"\nCannot allocate frame buffer in encoder.\n");
            }

            mpPacket->data = NULL;
            mpPacket->size = 0;

            int ret = 0;
            if (mHybridMode) {
                if (mpSwsContext) {
                    // We do color conversion if hybrid rendering and context exists
                    const int stride[1] = { mpCodecContext->width * 4 };
                    const uint8_t* const pData[1] = { sourceBuffer };
                    ret = sws_scale(mpSwsContext, pData, stride, 0, mpCodecContext->height, mpFrame->data, mpFrame->linesize);
                    if (ret < 0) {
                        av_strerror(ret, msg, 100);
                        OutputDebugString(L"\nCompression: Failure to color convert due to: ");
                        OutputDebugStringA(msg);
                        continue;
                    }
                }
                else { 
                    // No color conversion, so we have to manually load the data into the frame
                    // Future consideration: Have a function to copy data for each different pixel format.
                    // Currently used only for AO
                    for (int i = 0; i < AO_TEX_LEN / 4; i++) {
                        mpFrame->data[0][i] = sourceBuffer[i << 2];
                    }
                }
            }
            else {
                // We have to copy the values in our input into the frame
                //mpFrame->data[0] = &sourceBuffer[0];                         ///<  Y channel
                //mpFrame->data[1] = &sourceBuffer[nWidth * nHeight];          ///<  U channel
                //mpFrame->data[2] = &sourceBuffer[2 * nWidth * nHeight];      ///<  V channel
                for (int j = 0; j < nSize; j+=4) {
                    // Due to different endianness, texture: YUV0, buffer: 0VUY
                    int k = j >> 2;
                    mpFrame->data[2][k] = sourceBuffer[j + 1];
                    mpFrame->data[1][k] = sourceBuffer[j + 2];
                    mpFrame->data[0][k] = sourceBuffer[j + 3];
                }
            }

            /* Send frame to encoder and receive encoded packet*/
            ret = avcodec_send_frame(mpCodecContext, mpFrame);
            if (ret < 0) {
                av_strerror(ret, msg, 100);
                OutputDebugString(L"\nFailure to send frame due to: ");
                OutputDebugStringA(msg);
                continue;
            }

            ret = avcodec_receive_packet(mpCodecContext, mpPacket);
            if (ret == AVERROR(EAGAIN) || ret == AVERROR_EOF) {
                av_strerror(ret, msg, 100);
                OutputDebugString(L"\nNot enough data to compress. Waiting for more data. ");
                compressedSizes.push_back(0);
            }
            else if (ret < 0) {
                av_strerror(ret, msg, 100);
                OutputDebugString(L"\nCompression: Failure to receive packet due to: ");
                OutputDebugStringA(msg);
                compressedSizes.push_back(0);
            }

            /* Update size of compressed buffer */
            int compressedSize = mpPacket->size;
            outputBufferSize += compressedSize;

            /* Packet contains output buffer data*/
            memcpy(pOutputBuffer, mpPacket->data, mpPacket->size);
            /* Move the output buffer pointer to store next compressed buffer */
            pOutputBuffer += compressedSize;
            compressedSizes.push_back(compressedSize);

            char buffer[140];
            sprintf(buffer, "\n\n= Compressed Buffer: Original size: %d, Compressed size: %d =========", sourceBufferSize, compressedSize);
            OutputDebugStringA(buffer);

            /* Reset the frame and packet to be reused in the next loop */
            av_frame_unref(mpFrame);
            av_packet_unref(mpPacket);
        }

        // We store the sizes of the compressed buffers so the decoder can retrieve each one individually.
        for (int i = mNumOfTextures - 1; i >= 0; i--) {
            int bufSize = compressedSizes[i];
            /*
            * We store the ints using 4 bytes
            *    0     1     2     3 
            * | MSB | ... | ... | LSB | 
            */
            pOutputBuffer[0] = bufSize >> 24;
            pOutputBuffer[1] = bufSize >> 16 & 255;
            pOutputBuffer[2] = bufSize >> 8 & 255;
            pOutputBuffer[3] = bufSize & 255;
            pOutputBuffer += 4;
            outputBufferSize += 4;
        }
    }
    else { 
    // ===================== DECOMPRESSION ===================== //
        // We start from the end of the buffer to retrieve the sizes
        if (mGetInputBufferSize() == 0) return;
        uint8_t* pSourceBuffer = reinterpret_cast<uint8_t*>(mGetInputBuffer()) + mGetInputBufferSize();
        std::vector<int> compressedSizes; // Stores compressed sizes of all buffers
        // We first retrieve the sizes of all compressed buffers
        for (int i = 0; i < mNumOfTextures; i++) {
            pSourceBuffer -= 4;
            int bufSize = (pSourceBuffer[0] << 24) + (pSourceBuffer[1] << 16) + (pSourceBuffer[2] << 8) + (pSourceBuffer[3]);
            compressedSizes.push_back(bufSize);
        }

        pSourceBuffer = reinterpret_cast<uint8_t*>(mGetInputBuffer());
        char* pOutputBuffer = outputBuffer;
        outputBufferSize = 0;

        // Loop over all textures, decompress each one
        for (int i = 0; i < mNumOfTextures; i++) {
            std::lock_guard lock(ClientNetworkManager::mMutexClientVisTexRead);

            // Parameters for Decompression
            uint8_t* pSourceBuffer = reinterpret_cast<uint8_t*>(mGetInputBuffer()) + (i == 0 ? 0 : compressedSizes[i - 1]);
            int sourceBufferSize = compressedSizes[i];
            mpCodecContext = mpCodecContexts[i];
            mpSwsContext = mpSwsContexts[i];

            if (sourceBufferSize == 0) {
                // We currently don't have any data yet. Wait for another pass.
                outputBufferSize = 0;
                continue;
            }

            mpFrame->format = mpCodecContext->pix_fmt;
            mpFrame->width = mpCodecContext->width;
            mpFrame->height = mpCodecContext->height;

            /* AVPacket data must be av_malloc'd data so we do a copy here. */
            uint8_t* tmp_data = (uint8_t*)av_malloc(sourceBufferSize);
            memcpy(tmp_data, pSourceBuffer, sourceBufferSize);
            av_packet_from_data(mpPacket, tmp_data, sourceBufferSize);

            /* Send the encoded packet and receive decoded frame */
            int ret = avcodec_send_packet(mpCodecContext, mpPacket);
            if (ret < 0) {
                OutputDebugString(L"\nFailure to send packet due to: ");
                av_strerror(ret, msg, 100);
                OutputDebugStringA(msg);
            }

            ret = avcodec_receive_frame(mpCodecContext, mpFrame);
            if (ret == AVERROR(EAGAIN) || ret == AVERROR_EOF) {
                av_strerror(ret, msg, 100);
                OutputDebugString(L"\nNot enough data to decompress. Waiting for more data. ");
                continue;
            }
            else if (ret < 0) {
                av_strerror(ret, msg, 100);
                OutputDebugString(L"\nDecompression: Failure to receive frame due to: ");
                OutputDebugStringA(msg);
                continue;
            }

            if (mHybridMode) {
                if (mpSwsContext) {
                    /* Convert YUV -> RGBA into the output buffer */
                    const int stride[] = { mpCodecContext->width * 4 };
                    uint8_t* const pData[1] = { (uint8_t*)pOutputBuffer };
                    ret = sws_scale(mpSwsContext, mpFrame->data, mpFrame->linesize,
                        0, mpCodecContext->height, pData, stride);
                    if (ret < 0) {
                        av_strerror(ret, msg, 100);
                        OutputDebugString(L"\nDecompression: Failure to color convert due to: ");
                        OutputDebugStringA(msg);
                    }
                }
                else {
                    for (int i = 0, j = 0; i < AO_TEX_LEN / 4; i++, j+=4) {
                        pOutputBuffer[j] = mpFrame->data[0][i];
                    }
                }
            }
            else {
                // We copy the YUV data into the output buffer
                for (int j = 0; j < nSize; j += 4) {
                    int k = j >> 2;
                    outputBuffer[j] = 0;
                    outputBuffer[j+1] = mpFrame->data[2][k];
                    outputBuffer[j+2] = mpFrame->data[1][k];
                    outputBuffer[j+3] = mpFrame->data[0][k];
                }
            }

            // Not sure of how to find out the decompressed size, so assume decompressed size is always the same as original.
            char buffer[140];
            sprintf(buffer, "\n\n= Compressed Buffer: Original size: %d, Decompressed size: %d =========", sourceBufferSize, mBufferSizes[i]);
            OutputDebugStringA(buffer);

            // Update size of decompressed buffer
            outputBufferSize += mBufferSizes[i];
            pOutputBuffer += mBufferSizes[i];

            /* Reset the frame and packet to be reused in the next loop */
            av_frame_unref(mpFrame);
            av_packet_unref(mpPacket);
        }
        int x = 0;
    }
}

void CompressionPass::renderGui(Gui::Window* pPassWindow)
{
    int dirty = 0;
    pPassWindow->dropdown("Codec Method", mDisplayableBuffers, mCodecType);

    // If any of our UI parameters changed, let the pipeline know we're doing something different next frame
    if (dirty) setRefreshFlag();
}
