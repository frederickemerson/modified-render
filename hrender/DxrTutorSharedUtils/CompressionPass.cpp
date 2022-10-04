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
    
    mDisplayableBuffers.push_back({ 0, "LZ4" });
    mDisplayableBuffers.push_back({ 1, "H264" });

    if (mMode == Mode::Decompression) {
        initialiseH264Decoder();
        //initialiseDecoder();
    }
    else if (mMode == Mode::Compression) {
        initialiseH264Encoder();
        //initialiseEncoder();
    };
    return true;
}

bool CompressionPass::initialiseH264Encoder() {
    AVCodec* h264enc;
    if (isUsingCPU) {
        h264enc = avcodec_find_encoder(AV_CODEC_ID_H264);
    }
    else {
        h264enc = avcodec_find_encoder_by_name("h264_nvenc");
    }
    
    mpCodecContext = avcodec_alloc_context3(h264enc);
    if (!mpCodecContext) OutputDebugString(L"\nError: Could not allocated codec context.\n");

    /* Set up parameters for the context */
    mpCodecContext->width = nWidth;
    mpCodecContext->height = nHeight;
    mpCodecContext->time_base = av_make_q(1, 90000);
    int64_t bitrate = 900000;
    mpCodecContext->bit_rate = bitrate;
    mpCodecContext->rc_buffer_size = (int)bitrate / 60;
    mpCodecContext->ticks_per_frame = 2;
    mpCodecContext->thread_count = 0; // 0 makes FFmpeg choose the optimal number
    mpCodecContext->thread_type = FF_THREAD_FRAME;
    mpCodecContext->gop_size = 999999;
    mpCodecContext->pix_fmt = AV_PIX_FMT_YUV420P;

    AVDictionary* param = nullptr;
    av_dict_set(&param, "qp", "0", 0);
    if (isUsingCPU) {
        av_dict_set(&param, "preset", "ultrafast", 0);
        av_dict_set(&param, "profile:v", "high", 0);
        av_dict_set(&param, "vsync", "0", 0);
        av_dict_set(&param, "tune", "zerolatency", 0);
    }
    else {
        av_dict_set(&param, "preset", "llhp", 0);
        av_dict_set(&param, "profile:v", "high", 0);
        //av_dict_set(&param, "rc", "cbr_ld_hq", 0);
        av_dict_set(&param, "vsync", "0", 0);
    }



    /* Open the context for encoding */
    int err = avcodec_open2(mpCodecContext, h264enc, &param);
    if (err < 0) {
        av_strerror(err, msg, 100);
        OutputDebugString(L"\nFailure to open encoder context due to: ");
        OutputDebugStringA(msg);
    }

    /* Initialise the color converter */
    mpSwsContext = sws_getContext(nWidth, nHeight, AV_PIX_FMT_RGBA,
        nWidth, nHeight, AV_PIX_FMT_YUV420P,
        SWS_POINT, nullptr, nullptr, nullptr);

    /* Initialise the AVFrame and AVPacket*/
    mpFrame = av_frame_alloc();
    mpPacket = av_packet_alloc();

    return true;
}

bool CompressionPass::initialiseH264Decoder() {
    AVCodec* h264dec = avcodec_find_decoder(AV_CODEC_ID_H264);
    mpCodecContext = avcodec_alloc_context3(h264dec);
    if (!mpCodecContext) OutputDebugString(L"\nError: Could not allocated codec context.\n");

    /* Set params for the context */
    mpCodecContext->width = nWidth;
    mpCodecContext->height = nHeight;
    mpCodecContext->ticks_per_frame = 2;
    mpCodecContext->thread_count = 0; // 0 makes FFmpeg choose the optimal number
    mpCodecContext->thread_type = FF_THREAD_FRAME;
    mpCodecContext->pix_fmt = AV_PIX_FMT_YUV420P;

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

    /* Initialise the color converter + upscaling*/
    mpSwsContext = sws_getContext(
        nWidth, nHeight, AV_PIX_FMT_YUV420P,
        nWidth, nHeight, AV_PIX_FMT_RGBA,
        SWS_BILINEAR, nullptr, nullptr, nullptr
    );

    /* Initialise the AVFrame and AVPacket*/
    mpFrame = av_frame_alloc();
    mpPacket = av_packet_alloc();

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
                int sourceBufferSize = RenderConfig::BufferTypeToSize(RenderConfig::mConfig[i].type);

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
                int maxDecompressedSize = RenderConfig::BufferTypeToSize(RenderConfig::mConfig[i].type);

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
        // Loop over all textures, compress each one
        for (int i = 0; i < RenderConfig::mConfig.size(); i++)
        {
            std::lock_guard lock(ServerNetworkManager::mMutexServerVisTexRead);

            // Parameters for Compression
            uint8_t* sourceBuffer = reinterpret_cast<uint8_t*>(mGetInputBuffer());
            int sourceBufferSize = RenderConfig::BufferTypeToSize(RenderConfig::mConfig[i].type);
            outputBufferSize = 0;

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

            /* Convert RGBA -> YUV420P, for encoder friendly format. */
            const int stride[1] = { nWidth * 4 };
            const uint8_t* const pData[1] = { sourceBuffer };
            int ret = sws_scale(mpSwsContext, pData, stride, 0, nHeight, mpFrame->data, mpFrame->linesize);
            /* Send frame to encoder and receive encoded packet*/
            ret = avcodec_send_frame(mpCodecContext, mpFrame);
            if (ret < 0) {
                av_strerror(ret, msg, 100);
                OutputDebugString(L"\nFailure to send frame due to: ");
                OutputDebugStringA(msg);
            }

            ret = avcodec_receive_packet(mpCodecContext, mpPacket);
            if (ret < 0) {
                // We currently don't have enough frames collated to receive a packet. Wait for another pass.
                outputBufferSize = 0;
                continue;
            }

            /* Update size of compressed buffer */
            int compressedSize = mpPacket->size;
            outputBufferSize = compressedSize;

            /* Packet contains output buffer data*/
            memcpy(outputBuffer, mpPacket->data, mpPacket->size);
            char buffer[140];
            sprintf(buffer, "\n\n= Compressed Buffer: Original size: %d, Compressed size: %d =========", sourceBufferSize, compressedSize);
            OutputDebugStringA(buffer);

            /* Reset the frame and packet to be reused in the next loop */
            av_frame_unref(mpFrame);
            av_packet_unref(mpPacket);
        }
    }
    else { // mMode == Mode::Decompression
      // Loop over all textures, compress each one
        for (int i = 0; i < RenderConfig::mConfig.size(); i++) {
            std::lock_guard lock(ClientNetworkManager::mMutexClientVisTexRead);

            // Parameters for Decompression
            uint8_t* sourceBuffer = reinterpret_cast<uint8_t*>(mGetInputBuffer());
            int sourceBufferSize = mGetInputBufferSize();

            if (sourceBufferSize == 0) {
                // We currently don't have any data yet. Wait for another pass.
                outputBufferSize = 0;
                continue;
            }

            int maxDecompressedSize = RenderConfig::BufferTypeToSize(RenderConfig::mConfig[i].type);
            if (sourceBufferSize == maxDecompressedSize) {
                OutputDebugString(L"Skipping decompression, texture didnt change");
                return;
            }

            mpFrame->format = mpCodecContext->pix_fmt;
            mpFrame->width = mpCodecContext->width;
            mpFrame->height = mpCodecContext->height;

            /* AVPacket data must be av_malloc'd data so we do a copy here. */
            uint8_t* tmp_data = (uint8_t*)av_malloc(sourceBufferSize);
            memcpy(tmp_data, sourceBuffer, sourceBufferSize);
            av_packet_from_data(mpPacket, tmp_data, sourceBufferSize);

            /* Send the encoded packet and receive decoded frame */
            int ret = avcodec_send_packet(mpCodecContext, mpPacket);
            if (ret < 0) {
                OutputDebugString(L"\nFailure to send packet due to: ");
                av_strerror(ret, msg, 100);
                OutputDebugStringA(msg);
            }

            ret = avcodec_receive_frame(mpCodecContext, mpFrame);
            if (ret < 0) {
                // We currently don't have enough packets collated to receive a frame. Wait for another pass.
                outputBufferSize = 0;
                continue;
            }

            /* Convert YUV420P -> RGBA into the output buffer */
            const int stride[] = { nWidth * 4 };
            uint8_t* const pData[1] = { (uint8_t*)outputBuffer };
            ret = sws_scale(mpSwsContext, mpFrame->data, mpFrame->linesize,
                                0, nHeight, pData, stride);
            if (ret < 0) {
                av_strerror(ret, msg, 100);
                OutputDebugString(L"\ncannot scale due to : ");
                OutputDebugStringA(msg);
            }
            //memcpy(outputBuffer, mpFrame->data[0], nSize);
            // Not sure of how to find out the decompressed size, so for now decompressed size is always the same as original
            char buffer[140];
            sprintf(buffer, "\n\n= Compressed Buffer: Original size: %d, Decompressed size: %d =========", sourceBufferSize, nSize);
            OutputDebugStringA(buffer);

            // Update size of decompressed buffer
            outputBufferSize = nSize;

            /* Reset the frame and packet to be reused in the next loop */
            av_frame_unref(mpFrame);
            av_packet_unref(mpPacket);
        }
    }
}

void CompressionPass::renderGui(Gui::Window* pPassWindow)
{
    int dirty = 0;
    pPassWindow->dropdown("Codec Method", mDisplayableBuffers, mCodecType);

    // If any of our UI parameters changed, let the pipeline know we're doing something different next frame
    if (dirty) setRefreshFlag();
}
