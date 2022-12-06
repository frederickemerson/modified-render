#pragma once

#include "lz4.h"
#include "../DxrTutorSharedUtils/RenderPass.h"
#include "Interface/nvcuvid.h"
//#include "Interface/cuviddec.h"
#include "Interface/nvEncodeAPI.h"
#include "Samples/NvCodec/NvEncoder/NvEncoderD3D12.h"
#include "Samples/Utils/NvEncoderCLIOptions.h"
#include "Samples/NvCodec/NvEncoder/NvEncoderCuda.h"
#include "NvDecoder.h"
#include "cuda.h"
#include <wrl.h>
#include "../Utils/FFmpegDemuxer.h"

extern "C"
{
#include "libswscale/swscale.h"
}

class UploadInput
{
public:
    UploadInput(ID3D12Device* pDev, unsigned int numBfrs, unsigned int uploadBfrSize, unsigned int width, unsigned int height, NV_ENC_BUFFER_FORMAT bfrFormat)
    {
        pDevice = pDev;
        nWidth = width;
        nHeight = height;
        bufferFormat = bfrFormat;
        nInpBfrs = numBfrs;
        nCurIdx = 0;

        nFrameSize = nWidth * nHeight * 4;
        pHostFrame = std::unique_ptr<char[]>(new char[nFrameSize]);

        AllocateUploadBuffers(uploadBfrSize, nInpBfrs);

        D3D12_COMMAND_QUEUE_DESC gfxCommandQueueDesc = { D3D12_COMMAND_LIST_TYPE_DIRECT };
        if (pDevice->CreateCommandQueue(&gfxCommandQueueDesc, IID_PPV_ARGS(&pGfxCommandQueue)) != S_OK)
        {
            NVENC_THROW_ERROR("Failed to create command queue", NV_ENC_ERR_OUT_OF_MEMORY);
        }

        vCmdAlloc.resize(numBfrs);
        for (unsigned int i = 0; i < numBfrs; ++i)
        {
            if (pDevice->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&vCmdAlloc[i])) != S_OK)
            {
                NVENC_THROW_ERROR("Failed to create command allocator", NV_ENC_ERR_OUT_OF_MEMORY);
            }
        }

        if (pDevice->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, vCmdAlloc[0].Get(), nullptr, IID_PPV_ARGS(&pGfxCommandList)) != S_OK)
        {
            NVENC_THROW_ERROR("Failed to create command list", NV_ENC_ERR_OUT_OF_MEMORY);
        }

        if (pGfxCommandList->Close() != S_OK)
        {
            NVENC_THROW_ERROR("Failed to create command queue", NV_ENC_ERR_OUT_OF_MEMORY);
        }
    }

    ~UploadInput() {}

    void AllocateUploadBuffers(unsigned int uploadBfrSize, unsigned int numBfrs)
    {
        D3D12_HEAP_PROPERTIES heapProps{};
        heapProps.Type = D3D12_HEAP_TYPE_UPLOAD;
        heapProps.CPUPageProperty = D3D12_CPU_PAGE_PROPERTY_UNKNOWN;
        heapProps.MemoryPoolPreference = D3D12_MEMORY_POOL_UNKNOWN;

        D3D12_RESOURCE_DESC resourceDesc{};
        resourceDesc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
        resourceDesc.Alignment = 0;
        resourceDesc.Width = uploadBfrSize;
        resourceDesc.Height = 1;
        resourceDesc.DepthOrArraySize = 1;
        resourceDesc.MipLevels = 1;
        resourceDesc.Format = DXGI_FORMAT_UNKNOWN;
        resourceDesc.SampleDesc.Count = 1;
        resourceDesc.SampleDesc.Quality = 0;
        resourceDesc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
        resourceDesc.Flags = D3D12_RESOURCE_FLAG_NONE;

        vUploadBfr.resize(numBfrs);
        for (unsigned int i = 0; i < numBfrs; i++)
        {
            if (pDevice->CreateCommittedResource(&heapProps, D3D12_HEAP_FLAG_NONE, &resourceDesc, D3D12_RESOURCE_STATE_GENERIC_READ, nullptr,
                IID_PPV_ARGS(&vUploadBfr[i])) != S_OK)
            {
                NVENC_THROW_ERROR("Failed to create upload buffer", NV_ENC_ERR_OUT_OF_MEMORY);
            }
        }
    }

    void CopyToTexture(const NvEncInputFrame* encoderInputFrame, ID3D12Resource* pUploadBfr, NV_ENC_FENCE_POINT_D3D12* pInpFencePoint)
    {
        ID3D12Resource* pRsrc = (ID3D12Resource*)encoderInputFrame->inputPtr;
        ID3D12CommandAllocator* pGfxCommandAllocator = vCmdAlloc[nCurIdx % nInpBfrs].Get();
        D3D12_RESOURCE_DESC desc = pRsrc->GetDesc();
        D3D12_PLACED_SUBRESOURCE_FOOTPRINT inputUploadFootprint[2];

        pDevice->GetCopyableFootprints(&desc, 0, 1, 0, inputUploadFootprint, nullptr, nullptr, nullptr);

        if (pGfxCommandAllocator->Reset() != S_OK)
            NVENC_THROW_ERROR("Failed to reset command allocator", NV_ENC_ERR_OUT_OF_MEMORY);

        if (pGfxCommandList->Reset(pGfxCommandAllocator, NULL) != S_OK)
            NVENC_THROW_ERROR("Failed to reset command list", NV_ENC_ERR_OUT_OF_MEMORY);

        D3D12_RESOURCE_BARRIER barrier{};
        memset(&barrier, 0, sizeof(barrier));
        barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
        barrier.Flags = D3D12_RESOURCE_BARRIER_FLAG_NONE;
        barrier.Transition.pResource = pRsrc;
        barrier.Transition.StateBefore = D3D12_RESOURCE_STATE_COMMON;
        barrier.Transition.StateAfter = D3D12_RESOURCE_STATE_COPY_DEST;
        barrier.Transition.Subresource = 0;

        pGfxCommandList->ResourceBarrier(1, &barrier);

        {
            D3D12_TEXTURE_COPY_LOCATION copyDst{};
            copyDst.pResource = pRsrc;
            copyDst.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
            copyDst.SubresourceIndex = 0;

            D3D12_TEXTURE_COPY_LOCATION copySrc{};
            copySrc.pResource = pUploadBfr;
            copySrc.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
            copySrc.PlacedFootprint = inputUploadFootprint[0];

            pGfxCommandList->CopyTextureRegion(&copyDst, 0, 0, 0, &copySrc, nullptr);
        }

        memset(&barrier, 0, sizeof(barrier));
        barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
        barrier.Flags = D3D12_RESOURCE_BARRIER_FLAG_NONE;
        barrier.Transition.pResource = pRsrc;
        barrier.Transition.StateBefore = D3D12_RESOURCE_STATE_COPY_DEST;
        barrier.Transition.StateAfter = D3D12_RESOURCE_STATE_COMMON;
        barrier.Transition.Subresource = 0;

        pGfxCommandList->ResourceBarrier(1, &barrier);

        if (pGfxCommandList->Close() != S_OK)
            NVENC_THROW_ERROR("Failed to close command list", NV_ENC_ERR_OUT_OF_MEMORY);

        ID3D12CommandList* const ppCommandList[] = { pGfxCommandList.Get() };

        pGfxCommandQueue->ExecuteCommandLists(1, ppCommandList);

        InterlockedIncrement(&pInpFencePoint->value);

        // Signal fence from GPU side, encode will wait on this fence before reading the input
        pGfxCommandQueue->Signal((ID3D12Fence*)pInpFencePoint->pFence, pInpFencePoint->value);
    }

    bool ReadInputFrame(char* src, const NvEncInputFrame* encoderInputFrame, NV_ENC_FENCE_POINT_D3D12* pInpFencePoint)
    {
        //std::streamsize nRead = fpBgra.read(pHostFrame.get(), nFrameSize).gcount();

        ID3D12Resource* pUploadBfr = vUploadBfr[nCurIdx % nInpBfrs].Get();

        void* pData = nullptr;
        ck(pUploadBfr->Map(0, nullptr, &pData));

        char* pDst = (char*)pData;
        char* pSrc = src;
        unsigned int pitch = encoderInputFrame->pitch;
        char msg[60];
        sprintf(msg, "\nPitch: %u, nHeight: %u, nWidth: %u\n", pitch, nHeight, nWidth);
        OutputDebugStringA(msg);
        for (unsigned int y = 0; y < nHeight; y++)
        {
            memcpy(pDst + y * pitch, pSrc + y * nWidth * 4, nWidth * 4);
        }

        pUploadBfr->Unmap(0, nullptr);

        CopyToTexture(encoderInputFrame, pUploadBfr, pInpFencePoint);

        nCurIdx++;
        return true;
    }

private:
    ID3D12Device* pDevice;
    unsigned int nWidth, nHeight, nFrameSize;
    unsigned int nInpBfrs, nCurIdx;
    std::unique_ptr<char[]> pHostFrame;
    NV_ENC_BUFFER_FORMAT bufferFormat;

    Microsoft::WRL::ComPtr<ID3D12GraphicsCommandList> pGfxCommandList;
    Microsoft::WRL::ComPtr<ID3D12CommandQueue> pGfxCommandQueue;

    std::vector<Microsoft::WRL::ComPtr<ID3D12CommandAllocator>> vCmdAlloc;
    std::vector<Microsoft::WRL::ComPtr<ID3D12Resource>> vUploadBfr;
};


// Contains parameters for H264 encoding/decoding a texture
struct CodecParams {
    AVCodecID codecID;
    char* codecName; // Used when ID not available
    AVPixelFormat inPixFmt; // Source buffer pixel format
    AVPixelFormat outPixFmt; // Output buffer pixel format
    bool isColorConvertNeeded;  // Makes use of dstPixFmt if true.
    int width;
    int height;
};

/**
 * Transfer data from server to client or client to server
 * based on the configuration setting.
 */
class CompressionPass : public ::RenderPass
{

public:
    enum class Mode
    {
        Compression = 0,
        Decompression = 1
    };
    using SharedPtr = std::shared_ptr<CompressionPass>;
    using SharedConstPtr = std::shared_ptr<const CompressionPass>;
    virtual ~CompressionPass() = default;

    // Function for getting input buffers
    std::function<char* ()> mGetInputBuffer;
    std::function<int()> mGetInputBufferSize;

    static SharedPtr create(Mode mode, std::function<char* ()> getInputBuffer, std::function<int()> getInputBufferSize, bool isHybridRendering) {
        if (mode == Mode::Compression) {
            return SharedPtr(new CompressionPass(mode, getInputBuffer, "Compression Pass", "Compression Pass Gui", isHybridRendering));
        }
        else {
            return SharedPtr(new CompressionPass(mode, getInputBuffer, getInputBufferSize, "Decompression Pass", "Decompression Pass Gui", isHybridRendering));
        }
    }
    char* getOutputBuffer() { return outputBuffer; }
    int getOutputBufferSize() { return outputBufferSize; }

protected:
    CompressionPass(Mode mode, std::function<char* ()> getInputBuffer, const std::string name = "<Unknown render pass>",
        const std::string guiName = "<Unknown gui group>", bool isHybridRendering = true) :RenderPass(name, guiName) {
        mMode = mode;
        mGetInputBuffer = getInputBuffer;
        mHybridMode = isHybridRendering;
    }

    CompressionPass(Mode mode, std::function<char* ()> getInputBuffer, std::function<int()> getInputBufferSize,
        const std::string name = "<Unknown render pass>", const std::string guiName = "<Unknown gui group>", bool isHybridRendering = true) :RenderPass(name, guiName) {
        mMode = mode;
        mGetInputBuffer = getInputBuffer;
        mGetInputBufferSize = getInputBufferSize;
        mHybridMode = isHybridRendering;
    }

    // Buffer for storing output of compression/decompression
    int outputBufferSize;
    char* intermediateBuffer;
    char* outputBuffer;
    char* outputBufferNVENC;

    // Implementation of RenderPass interface
    bool initialize(RenderContext* pRenderContext, ResourceManager::SharedPtr pResManager) override;
    bool initialiseEncoder();

    bool initialiseH264Encoders();
    bool initialiseH264Decoders();

    bool initialiseH264HybridEncoder(CodecParams* codecParams);
    bool initialiseH264HybridDecoder(CodecParams* codecParams);
    bool initialiseH264RemoteEncoder();
    bool initialiseH264RemoteDecoder();
    bool initialiseDecoder();
    bool initialiseDecoder2();
    void DecodeMediaFile();
    void initScene(RenderContext* pRenderContext, Scene::SharedPtr pScene) override;
    void execute(RenderContext* pRenderContext) override;
    void executeLZ4(RenderContext* pRenderContext);
    void executeH264(RenderContext* pRenderContext);
    void executeNVENC(RenderContext* pRenderContext);
    void renderGui(Gui::Window* pPassWindow) override;

    Mode                                    mMode;                     ///< Whether this pass runs as compression or decompression

    CUdevice cuDevice = 0;
    CUcontext cuContext = NULL;
    CUvideodecoder m_hDecoder;

    FFmpegDemuxer* demuxer = nullptr;
    std::shared_ptr<NvDecoder> dec;
    int nVideoBytes = 0, nFrameReturned = 0, nFrame = 0;
    uint8_t* pFrame;
    uint8_t* pVideo = NULL;

    Microsoft::WRL::ComPtr<ID3D12Device> pDevice;
    Microsoft::WRL::ComPtr<IDXGIFactory1> pFactory;
    Microsoft::WRL::ComPtr<IDXGIAdapter> pAdapter;
    Microsoft::WRL::ComPtr<ID3D12Debug> debugController;
    DXGI_ADAPTER_DESC adapterDesc;
    std::shared_ptr<NvEncoderD3D12> enc;
    NV_ENC_BUFFER_FORMAT bufferFormat = NV_ENC_BUFFER_FORMAT_ARGB;
    std::unique_ptr<UploadInput> pUploadInput;
    const unsigned int pSliceDataOffset = 0; // for CUVIDPICPARAMS
    int iGpu = 0;
    int nWidth = 1920;
    int nHeight = 1080;
    int nSize = nWidth * nHeight * 4;
    char msg[100];

    // Each different buffer has different encoder settings.
    AVFrame* mpFrame = nullptr;
    AVPacket* mpPacket = nullptr;

    std::vector<struct SwsContext*> mpSwsContexts;
    std::vector<AVCodecContext*> mpCodecContexts;

    struct SwsContext* mpSwsContext = nullptr;
    AVCodecContext* mpCodecContext = nullptr;

    Gui::DropdownList mDisplayableBuffers;
    bool isUsingCPU = true;
    bool isRemoteRendering = false;                                ///< True if rendering whole scene on server
    uint32_t          mCodecType = H264;                           ///< H264 by default
    enum CodecType : uint32_t {
        LZ4,
        H264
    };
    bool mHybridMode = true;                                       ///< True if doing hybrid rendering, else remote rendering.
    int mNumOfTextures = 2;                                        ///< Number of textures to encode each frame
    int mBufferOffsets[2] = { 0, VIS_TEX_LEN };                    ///< Offset before next buffer
    int mBufferSizes[2] = { VIS_TEX_LEN, AO_TEX_LEN };             ///< Size of each buffer 
};