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
    JitteredGBufferPass,
    VisibilityPass,
    MemoryTransferPassGPU_CPU,
    MemoryTransferPassCPU_GPU,
    CompressionPass,
    DecompressionPass,
    NetworkClientRecvPass,
    NetworkClientSendPass,
    NetworkServerRecvPass,
    NetworkServerSendPass,
    VShadingPass,
    CopyToOutputPass,
    SimpleAccumulationPass,
    SimulateDelayPass,
    PredictionPass,
    ScreenSpaceReflectionPass,
    ServerRayTracingReflectionPass,
    ReflectionCompositePass
};


// total size: 16 bytes
struct RenderConfiguration {
    int texWidth;
    int texHeight;
    uint8_t sceneIndex;
    uint8_t numPasses; // 1 byte
    RenderConfigPass passOrder[17]; // 1 * 17 bytes
};