#include "RenderConfigUtils.h"

const int texWidth = 1920;
const int texHeight = 1080;

/*
* Currently, reflections are disabled. GI is also not functional at the moment.
* It was found that since reflections send full colour data over the network, we need a virtually lossless
* network to sustain the interactivity of the application.
* 
* For GI, it is the same issue of sending colour data over the network, more specifically the indirect
* colour has to be sent over the network.
* 
* You may reenable reflections by uncommenting the commented portions below, and changing the size of 
* the vector passed in hrender.cpp. GI may have to be debugged to run properly.
*/
RenderConfiguration getDebugRenderConfig(RenderMode mode, RenderType type, unsigned char sceneIdx) {
	if (mode == RenderMode::HybridRender) {
		if (type == RenderType::Distributed) {
            return {
                texWidth, texHeight, // texWidth and texHeight
                sceneIdx, // sceneIndex
                13,
                { // Array of RenderConfigPass
                    JitteredGBufferPass,
                    VisibilityPass,
                    AmbientOcclusionPass,
                    //ScreenSpaceReflectionPass,
                    //ServerRayTracingReflectionPass,
                    DistrSVGFPass,
                    MemoryTransferPassGPU_CPU,
                    CompressionPass,
                    SimulateDelayPass,
                    DecompressionPass,
                    MemoryTransferPassCPU_GPU,
                    PredictionPass,
                    VShadingPass,
                    //ReflectionCompositePass,
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
                17,
                { // Array of RenderConfigPass
                    JitteredGBufferPass,
                    VisibilityPass,
                    AmbientOcclusionPass,
                    ScreenSpaceReflectionPass,
                    ServerRayTracingReflectionPass,
                    PredictionPass,
                    VShadingPass,
                    ReflectionCompositePass,
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
                8,
                { // Array of RenderConfigPass
                    NetworkServerRecvPass,
                    JitteredGBufferPass,
                    VisibilityPass,
                    //ScreenSpaceReflectionPass,
                    //ServerRayTracingReflectionPass,
                    AmbientOcclusionPass,
                    DistrSVGFPass,
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
                    //ScreenSpaceReflectionPass,
                    //ServerRayTracingReflectionPass,
                    AmbientOcclusionPass,
                    PredictionPass,
                    VShadingPass,
                    //ReflectionCompositePass,
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
                    //ScreenSpaceReflectionPass,
                    VShadingPass,
                    //ReflectionCompositePass,
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

