#pragma once

#include <vector>
#include <string>
#include <map>
#include <iostream>

class RenderConfig {
public:
    static const int RENDER_CONFIGURATION_MAX_PASSES_SUPPORTED = 14;

    enum class BufferType {
        VisibilityBitmap
    };

    struct Config {
        BufferType type;
        std::string name; // name in ResourceManager
        int resourceIndex; // index in ResourceManager
        int compressedSize;
        int compressedSize2;
    };

    static void setConfiguration(std::vector<BufferType> orderedTypes);
    static std::string print();
    static int getTotalSize();
    static int BufferTypeToSize(RenderConfig::BufferType htype);

    static std::vector<Config> getConfig();
    static std::vector<Config> mConfig;

protected:
    static int totalSize;
    static std::string BufferTypeToString(BufferType htype);
};

enum RenderConfigPass : uint8_t {
    // --- JitteredGBufferPass creates a GBuffer --- //
    JitteredGBufferPass,
    // --- VisibilityPass makes use of the GBuffer determining visibility under different lights --- //
    VisibilityPass,
    // --- MemoryTransferPassGPU_CPU transfers GPU information into CPU --- //
    MemoryTransferPassGPU_CPU,
    // --- MemoryTransferPassCPU_GPU transfers CPU information into GPU --- //
    MemoryTransferPassCPU_GPU,
    // --- CompressionPass compresses buffers to be sent across Network --- //
    CompressionPass,
    // --- DecompressionPass decompresses buffers sent across Network--- //
    DecompressionPass,
    // --- NetworkClientRecvPass receives visibility bitmap from server --- //
    NetworkClientRecvPass,
    // --- NetworkClientSendPass sends camera data to server --- //
    NetworkClientSendPass,
    // --- NetworkServerRecvPass receives camera data from client --- //
    NetworkServerRecvPass,
    // --- NetworkServerSendPass sends the visibility bitmap to the client --- //
    NetworkServerSendPass,
    // --- VShadingPass makes use of the direct and indirect lighting to shade the sceneIndex.
    //     We also provide the ability to preview the GBuffer alternatively. --- //
    VShadingPass,
    // --- CopyToOutputPass just lets us select which pass to view on screen --- //
    CopyToOutputPass,
    // --- SimpleAccumulationPass temporally accumulates frames for denoising --- //
    SimpleAccumulationPass,
    // --- SimulateDelayPass simulates delay across network --- //
    SimulateDelayPass,
    // --- PredictionPass performs prediction on visibility bitmap if frames are behind. --- //
    PredictionPass,
    // --- ServerRemoteConverter converts the final image output to R11G11B10 float format. --- //
    ServerRemoteConverter,
    // --- ClientRemoteConverter converts the R11G11B10 floats back to RGBA32Float format. --- //
    ClientRemoteConverter,
    // --- AmbientOcclusionPass performs per-pixel ambient occlusion --- //
    AmbientOcclusionPass,
    // --- ServerGlobalIllumPass computes indirect illumination color and
    //     selects random light index to be used for direct illumination. --- //
    ServerGlobalIllumPass,
    // --- ClientGlobalIllumPass loads server indirect illumination into texture and
    //     calculates direct illumination using given random light index --- //
    ClientGlobalIllumPass,
    // --- SVGFServerPass performs denoising on indirect illumination.
    SVGFServerPass,
    // --- SVGFClientPass performs denoising on direct illumination
    SVGFClientPass,
};

// total size: 16 bytes
struct RenderConfiguration {
    int texWidth;
    int texHeight;
    uint8_t sceneIndex;
    uint8_t numPasses; // 1 byte
    RenderConfigPass passOrder[14]; // 1 * 14 bytes
};

enum class RenderMode : uint8_t {
    RemoteRender, // Scenes fully rendered on server
    HybridRender
};

enum class RenderType : uint8_t {
    Distributed, // Works with distributed
    GGXGlobalIllum // Only works on debug mode
};
