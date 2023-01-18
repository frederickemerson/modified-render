#include "RenderConfigUtils.h"

const int texWidth = 1920;
const int texHeight = 1080;

RenderConfiguration getDebugRenderConfig(RenderMode mode, RenderType type, unsigned char sceneIdx) {
	if (mode == RenderMode::HybridRender) {
		if (type == RenderType::Distributed) {
            return {
                texWidth, texHeight, // texWidth and texHeight
                sceneIdx, // sceneIndex
                12,
                { // Array of RenderConfigPass
                    JitteredGBufferPass,
                    VisibilityPass,
                    AmbientOcclusionPass,
                    MemoryTransferPassGPU_CPU,
                    CompressionPass,
                    SimulateDelayPass,
                    DecompressionPass,
                    MemoryTransferPassCPU_GPU,
                    PredictionPass,
                    VShadingPass,
                    CopyToOutputPass,
                    SimpleAccumulationPass
                }
            };
        } else { // GGXGlobalIllum
            return {
                texWidth, texHeight, // texWidth and texHeight
                sceneIdx, // sceneIndex
                12,
                { // Array of RenderConfigPass
                    JitteredGBufferPass,
                    ServerGlobalIllumPass,
                    SVGFServerPass,
                    MemoryTransferPassGPU_CPU,
                    CompressionPass,
                    SimulateDelayPass,
                    DecompressionPass,
                    MemoryTransferPassCPU_GPU,
                    ClientGlobalIllumPass,
                    SVGFClientPass,
                    CopyToOutputPass,
                    SimpleAccumulationPass
                }
            };
        }
    }
    else { // Remote rendering
        if (type == RenderType::Distributed) {
            return {
                texWidth, texHeight, // texWidth and texHeight
                sceneIdx, // sceneIndex
                14,
                { // Array of RenderConfigPass
                    JitteredGBufferPass,
                    VisibilityPass,
                    AmbientOcclusionPass,
                    PredictionPass,
                    VShadingPass,
                    ServerRemoteConverter,
                    MemoryTransferPassGPU_CPU,
                    CompressionPass,
                    SimulateDelayPass,
                    DecompressionPass,
                    MemoryTransferPassCPU_GPU,
                    ClientRemoteConverter,
                    CopyToOutputPass,
                    SimpleAccumulationPass
                }
            };
        }
        else { // GGXGlobalIllum
            return {
                texWidth, texHeight, // texWidth and texHeight
                sceneIdx, // sceneIndex
                14,
                { // Array of RenderConfigPass
                    JitteredGBufferPass,
                    ServerGlobalIllumPass,
                    SVGFServerPass,
                    ClientGlobalIllumPass,
                    SVGFClientPass,
                    ServerRemoteConverter,
                    MemoryTransferPassGPU_CPU,
                    CompressionPass,
                    SimulateDelayPass,
                    DecompressionPass,
                    MemoryTransferPassCPU_GPU,
                    ClientRemoteConverter,
                    CopyToOutputPass,
                    SimpleAccumulationPass
                }
            };
        }      
    }
}

RenderConfiguration getServerRenderConfig(RenderMode mode, RenderType type, unsigned char sceneIdx) {
    if (mode == RenderMode::HybridRender) {
        if (type == RenderType::Distributed) {
            return {
                texWidth, texHeight, // texWidth and texHeight
                sceneIdx, // sceneIndex
                7,
                { // Array of RenderConfigPass
                    NetworkServerRecvPass,
                    JitteredGBufferPass,
                    VisibilityPass,
                    AmbientOcclusionPass,
                    MemoryTransferPassGPU_CPU,
                    CompressionPass,
                    NetworkServerSendPass
                }
            };
        }
        else { // GGXGlobalIllum
            return {
                texWidth, texHeight, // texWidth and texHeight
                sceneIdx, // sceneIndex
                7,
                { // Array of RenderConfigPass
                    NetworkServerRecvPass,
                    JitteredGBufferPass,
                    ServerGlobalIllumPass,
                    SVGFServerPass,
                    MemoryTransferPassGPU_CPU,
                    CompressionPass,
                    NetworkServerSendPass
                }
            };
        }
    }
    else { // Remote rendering
        if (type == RenderType::Distributed) {
            return {
                texWidth, texHeight, // texWidth and texHeight
                sceneIdx, // sceneIndex
                10,
                { // Array of RenderConfigPass
                    NetworkServerRecvPass,
                    JitteredGBufferPass,
                    VisibilityPass,
                    AmbientOcclusionPass,
                    PredictionPass,
                    VShadingPass,
                    ServerRemoteConverter,
                    MemoryTransferPassGPU_CPU,
                    CompressionPass,
                    NetworkServerSendPass
                }
            };
        }
        else { // GGXGlobalIllum
            return {
                texWidth, texHeight, // texWidth and texHeight
                sceneIdx, // sceneIndex
                10,
                { // Array of RenderConfigPass
                    NetworkServerRecvPass,
                    JitteredGBufferPass,
                    ServerGlobalIllumPass,
                    SVGFServerPass,
                    ClientGlobalIllumPass,
                    SVGFClientPass,
                    ServerRemoteConverter,
                    MemoryTransferPassGPU_CPU,
                    CompressionPass,
                    NetworkServerSendPass
                }
            };
        }
    }
}

RenderConfiguration getClientRenderConfig(RenderMode mode, RenderType type, unsigned char sceneIdx) {
    if (mode == RenderMode::HybridRender) {
        if (type == RenderType::Distributed) {
            return {
                texWidth, texHeight, // texWidth and texHeight
                sceneIdx, // sceneIndex
                9,
                { // Array of RenderConfigPass
                    NetworkClientSendPass,
                    NetworkClientRecvPass,
                    DecompressionPass,
                    MemoryTransferPassCPU_GPU,
                    PredictionPass,
                    VShadingPass,
                    CopyToOutputPass,
                    SimpleAccumulationPass,
                    JitteredGBufferPass
                }
            };
        }
        else { // GGXGlobalIllum
            return {
                texWidth, texHeight, // texWidth and texHeight
                sceneIdx, // sceneIndex
                9,
                { // Array of RenderConfigPass
                    JitteredGBufferPass,
                    NetworkClientSendPass,
                    NetworkClientRecvPass,
                    DecompressionPass,
                    MemoryTransferPassCPU_GPU,
                    ClientGlobalIllumPass,
                    SVGFClientPass,
                    CopyToOutputPass,
                    SimpleAccumulationPass
                }
            };
        }
    }
    else { // Remote rendering
        return {
            texWidth, texHeight, // texWidth and texHeight
            sceneIdx, // sceneIndex
            8,
            { // Array of RenderConfigPass
                NetworkClientSendPass,
                NetworkClientRecvPass,
                DecompressionPass,
                MemoryTransferPassCPU_GPU,
                ClientRemoteConverter,
                CopyToOutputPass,
                SimpleAccumulationPass,
                JitteredGBufferPass
            }
        };
    }
}

