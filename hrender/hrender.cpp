/***************************************************************************
 # Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
 #
 # Redistribution and use in source and binary forms, with or without
 # modification, are permitted provided that the following conditions
 # are met:
 #  * Redistributions of source code must retain the above copyright
 #    notice, this list of conditions and the following disclaimer.
 #  * Redistributions in binary form must reproduce the above copyright
 #    notice, this list of conditions and the following disclaimer in the
 #    documentation and/or other materials provided with the distribution.
 #  * Neither the name of NVIDIA CORPORATION nor the names of its
 #    contributors may be used to endorse or promote products derived
 #    from this software without specific prior written permission.
 #
 # THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 # EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 # IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 # PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 # CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 # EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 # PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 # PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 # OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 # (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 # OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 **************************************************************************/
#include "hrender.h"
#include "DxrTutorSharedUtils/RenderConfigUtils.h"
#include "DxrTutorCommonPasses/CopyToOutputPass.h"
#include "TestPasses/DecodeGBufferPass.h"
#include "DxrTutorCommonPasses/JitteredGBufferPass.h"
#include "DxrTutorCommonPasses/LambertianPlusShadowPass.h"
#include "DxrTutorCommonPasses/SimpleAccumulationPass.h"
#include "DxrTutorSharedUtils/RenderingPipeline.h"
#include "NetworkPasses/VisibilityPass.h"
#include "NetworkPasses/PredictionPass.h"
#include "NetworkPasses/VShadingPass.h"
#include "NetworkPasses/SimulateDelayPass.h"
#include "NetworkPasses/MemoryTransferPassClientCPU_GPU.h"
#include "NetworkPasses/MemoryTransferPassServerGPU_CPU.h"
#include "NetworkPasses/NetworkClientRecvPass.h"
#include "NetworkPasses/NetworkClientSendPass.h"
#include "NetworkPasses/NetworkServerRecvPass.h"
#include "NetworkPasses/NetworkServerSendPass.h"
#include "DxrTutorSharedUtils/CompressionPass.h"
#include "NetworkPasses/PredictionPass.h"
#include "DxrTutorCommonPasses/ServerRemoteConverter.h"
#include "DxrTutorCommonPasses/ClientRemoteConverter.h"
#include "DxrTutorCommonPasses/AmbientOcclusionPass.h"
#include "DxrTutorCommonPasses/GGXClientGlobalIllumPass.h"
#include "DxrTutorCommonPasses/GGXServerGlobalIllumPass.h"
#include "SVGFPasses/SVGFServerPass.h"
#include "SVGFPasses/SVGFClientPass.h"
#include "SVGFPasses/DistrSVGFPass.h"
#include "NetworkPasses/ScreenSpaceReflectionPass.h"
#include "NetworkPasses/ServerRayTracingReflectionPass.h"
#include "NetworkPasses/ReflectionCompositePass.h"


void runServer();
void runClient();
void runDebug();

// Available scenes and corresponding environment maps
std::string defaultSceneNames[] = {
    "pink_room\\pink_room.pyscene", // PINK ROOM //
    "SunTemple\\SunTemple_v3.pyscene", // SUN TEMPLE //
    "Bistro\\BistroInterior_v4.pyscene", // BISTRO //
    "ZeroDay\\MEASURE_ONE\\MEASURE_ONE.fbx" // ZERO DAY //
};

const char* environmentMaps[] = {
    "MonValley_G_DirtRoad_3k.hdr", // PINK ROOM //
    "SunTemple\\SunTemple_Skybox.hdr", // SUN TEMPLE //
    "Bistro\\san_giuseppe_bridge_4k.hdr", // BISTRO //
    "MonValley_G_DirtRoad_3k.hdr" // ZERO DAY //
};

// Scene index follows the scenes mentioned above.
unsigned char sceneIdx = 0;

// Switches rendering modes between HybridRender and RemoteRender
RenderMode renderMode = RenderMode::HybridRender;
//RenderMode renderMode = RenderMode::RemoteRender;

// GGXGlobalIllum serves as ground truth; only works on debug mode.
RenderType renderType = RenderType::Distributed;
//RenderType renderType = RenderType::GGXGlobalIllum;

/**
 * Determines the mode or configuration that the program runs
 * based on the command line argument.
 */
int WINAPI WinMain(_In_ HINSTANCE hInstance, _In_opt_ HINSTANCE hPrevInstance, _In_ LPSTR lpCmdLine, _In_ int nShowCmd)
{
    if (std::string(lpCmdLine).find(std::string("no-compression")) != std::string::npos)
    {
        OutputDebugString(L"\n\n======== WITHOUT COMPRESSION =========");
    }

    if (std::string(lpCmdLine).find(std::string("server")) != std::string::npos)
    {
        OutputDebugString(L"\n\n======== SERVER MODE =========");
        runServer();
    }
    else if (std::string(lpCmdLine).find(std::string("client")) != std::string::npos)
    {
        OutputDebugString(L"\n\n======== CLIENT MODE =========");
        runClient();
    }
    else
    {
        OutputDebugString(L"\n\n======== DEBUG MODE =========");
        runDebug();
    }
    return 0;
}

void CreatePipeline(RenderConfiguration renderConfiguration, RenderingPipeline* pipeline) {
    pipeline->setDefaultSceneName(defaultSceneNames[renderConfiguration.sceneIndex]);
    pipeline->updateEnvironmentMap(environmentMaps[renderConfiguration.sceneIndex]);

    if (renderConfiguration.numPasses <= 0 ||
        renderConfiguration.numPasses > RenderConfig::RENDER_CONFIGURATION_MAX_PASSES_SUPPORTED) {
        OutputDebugString(L"Error: incorrect num passes parameter in render configuration");
    }

    std::function<char*()> inputBufferArgument;
    std::function<int ()> inputBufferSizeArgument;
    bool isHybridRendering = renderMode == RenderMode::HybridRender;

    for (int i = 0; i < renderConfiguration.numPasses; i++) {
        if (renderConfiguration.passOrder[i] == JitteredGBufferPass) {
            pipeline->setPass(i, JitteredGBufferPass::create(renderConfiguration.texWidth, renderConfiguration.texHeight));
        }
        else if (renderConfiguration.passOrder[i] == VisibilityPass) {
            pipeline->setPass(i, VisibilityPass::create("VisibilityBitmap", "WorldPosition", renderConfiguration.texWidth, renderConfiguration.texHeight));
        }
        else if (renderConfiguration.passOrder[i] == MemoryTransferPassGPU_CPU) {
            auto pass = MemoryTransferPassServerGPU_CPU::create(isHybridRendering);
            pipeline->setPass(i, pass);
            inputBufferArgument = std::bind(&MemoryTransferPassServerGPU_CPU::getOutputBuffer, pass.get());
            inputBufferSizeArgument = std::bind(&MemoryTransferPassServerGPU_CPU::getOutputBufferSize, pass.get());
        }
        else if (renderConfiguration.passOrder[i] == MemoryTransferPassCPU_GPU) {
            pipeline->setPass(i, MemoryTransferPassClientCPU_GPU::create(inputBufferArgument, isHybridRendering));
        }
        else if (renderConfiguration.passOrder[i] == CompressionPass) {
            auto pass = CompressionPass::create(CompressionPass::Mode::Compression, inputBufferArgument, inputBufferSizeArgument, isHybridRendering);
            pipeline->setPass(i, pass);
            inputBufferArgument = std::bind(&CompressionPass::getOutputBuffer, pass.get());
            inputBufferSizeArgument = std::bind(&CompressionPass::getOutputBufferSize, pass.get());
        }
        else if (renderConfiguration.passOrder[i] == DecompressionPass) {
            auto pass = CompressionPass::create(CompressionPass::Mode::Decompression, inputBufferArgument, inputBufferSizeArgument, isHybridRendering);
            pipeline->setPass(i, pass);
            inputBufferArgument = std::bind(&CompressionPass::getOutputBuffer, pass.get());
        }
        else if (renderConfiguration.passOrder[i] == NetworkClientRecvPass) {
            pipeline->setPass(i, NetworkClientRecvPass::create(renderConfiguration.texWidth, renderConfiguration.texHeight));
            inputBufferArgument = std::bind(&ClientNetworkManager::getOutputBuffer, ResourceManager::mClientNetworkManager.get());
            inputBufferSizeArgument = std::bind(&ClientNetworkManager::getOutputBufferSize, ResourceManager::mClientNetworkManager.get());
        }
        else if (renderConfiguration.passOrder[i] == NetworkClientSendPass) {
            pipeline->setPass(i, NetworkClientSendPass::create(renderConfiguration.texWidth, renderConfiguration.texHeight));
        }
        else if (renderConfiguration.passOrder[i] == NetworkServerRecvPass) {
            pipeline->setPass(i, NetworkServerRecvPass::create(renderConfiguration.texWidth, renderConfiguration.texHeight));
        }
        else if (renderConfiguration.passOrder[i] == NetworkServerSendPass) {
            pipeline->setPass(i, NetworkServerSendPass::create(renderConfiguration.texWidth, renderConfiguration.texHeight));
            ResourceManager::mServerNetworkManager->mGetInputBuffer = inputBufferArgument;
            ResourceManager::mServerNetworkManager->mGetInputBufferSize = inputBufferSizeArgument;
        }
        else if (renderConfiguration.passOrder[i] == VShadingPass) {
            std::string outBuf = isHybridRendering ? "V-shading" : "__V-shadingYUVServer";
            pipeline->setPassOptions(i, { 
                VShadingPass::create(outBuf, isHybridRendering),
                DecodeGBufferPass::create("DecodedGBuffer") 
            });
        }
        else if (renderConfiguration.passOrder[i] == CopyToOutputPass) {
            pipeline->setPass(i, CopyToOutputPass::create());
        }
        else if (renderConfiguration.passOrder[i] == SimpleAccumulationPass) {
            pipeline->setPass(i, SimpleAccumulationPass::create(ResourceManager::kOutputChannel));
        }
        else if (renderConfiguration.passOrder[i] == SimulateDelayPass) {
            auto pass = SimulateDelayPass::create(inputBufferArgument, inputBufferSizeArgument);
            pipeline->setPass(i, pass);
            inputBufferArgument = std::bind(&SimulateDelayPass::getOutputBuffer, pass.get());
            inputBufferSizeArgument = std::bind(&SimulateDelayPass::getOutputBufferSize, pass.get());
        }
        else if (renderConfiguration.passOrder[i] == PredictionPass) {
            auto pass = PredictionPass::create(renderConfiguration.texWidth, renderConfiguration.texHeight);
            pipeline->setPass(i, pass);
        }
        else if (renderConfiguration.passOrder[i] == ServerRemoteConverter) {
            // Simple pass to convert RGBA32Float to R11G11B10 floats
            pipeline->setPass(i, ServerRemoteConverter::create("RemoteIllum", "V-shadingServer", renderConfiguration.texWidth, renderConfiguration.texHeight));
        }
        else if (renderConfiguration.passOrder[i] == ClientRemoteConverter) {
            // Simple pass to convert R11G11B10 to RGBA32Float floats
            pipeline->setPass(i, ClientRemoteConverter::create("V-shadingClient", "V-shading"));
        }
        else if (renderConfiguration.passOrder[i] == AmbientOcclusionPass) {
            pipeline->setPass(i, AmbientOcclusionPass::create("AmbientOcclusion", renderConfiguration.texWidth, renderConfiguration.texHeight));
        }
        else if (renderConfiguration.passOrder[i] == ServerGlobalIllumPass) {
            pipeline->setPass(i, GGXServerGlobalIllumPass::create("OutIndirectColor", "OutIndirectAlbedo", renderConfiguration.texWidth, renderConfiguration.texHeight));
        }
        else if (renderConfiguration.passOrder[i] == ClientGlobalIllumPass) {
            pipeline->setPass(i, GGXClientGlobalIllumPass::create("OutDirectColor", "OutDirectAlbedo", renderConfiguration.texWidth, renderConfiguration.texHeight));
        }
        else if (renderConfiguration.passOrder[i] == SVGFServerPass) {
            pipeline->setPass(i, SVGFServerPass::create("OutIndirectColor", "ServerIndirectLighting", renderConfiguration.texWidth, renderConfiguration.texHeight));
        }
        else if (renderConfiguration.passOrder[i] == SVGFClientPass) {
            std::string outBuf = isHybridRendering ? "V-shading" : "RemoteIllum";
            pipeline->setPass(i, SVGFClientPass::create("OutDirectColor", outBuf, isHybridRendering));
        }
        else if (renderConfiguration.passOrder[i] == ScreenSpaceReflectionPass) {
            pipeline->setPass(i, ScreenSpaceReflectionPass::create());
        }
        else if (renderConfiguration.passOrder[i] == ServerRayTracingReflectionPass) {
            pipeline->setPass(i, ServerRayTracingReflectionPass::create("SRTReflection", renderConfiguration.texWidth, renderConfiguration.texHeight));
        }
        else if (renderConfiguration.passOrder[i] == ReflectionCompositePass) {
            pipeline->setPass(i, ReflectionCompositePass::create());
        }
        else if (renderConfiguration.passOrder[i] == DistrSVGFPass) {
            pipeline->setPass(i, DistrSVGFPass::create("VisibilityBitmap", "AmbientOcclusion", renderConfiguration.texWidth, renderConfiguration.texHeight));
        } 
    }
}

/**
 * Runs the program as a stand alone program (no server, no client).
 */
void runDebug()
{
    // hrender config
    RenderConfig::setConfiguration({ RenderConfig::BufferType::VisibilityBitmap, RenderConfig::BufferType::SRTReflection });

    // Define a set of mConfig / window parameters for our program
    SampleConfig config;
    config.windowDesc.title = "NRender";
    config.windowDesc.resizableWindow = true;

    RenderConfiguration renderConfiguration = getDebugRenderConfig(renderMode, renderType, sceneIdx);
    // Create our rendering pipeline
    RenderingPipeline* pipeline = new RenderingPipeline(true, uint2(renderConfiguration.texWidth, renderConfiguration.texHeight));

    CreatePipeline(renderConfiguration, pipeline);

    // ============================ //
    // Set presets for the pipeline //
    // ============================ //
    if (renderMode == RenderMode::HybridRender) {
        pipeline->setPresets({
            RenderingPipeline::PresetData("Regular shading", "V-shading", { 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 }),
            RenderingPipeline::PresetData("Preview GBuffer", "DecodedGBuffer", { 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1 }),
            RenderingPipeline::PresetData("No compression, no memory transfer", "V-shading", { 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1 })
            });
    }
    else if (renderMode == RenderMode::RemoteRender) {
        pipeline->setPresets({
            RenderingPipeline::PresetData("Regular shading", "V-shading", { 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 }),
            });
    }

    // Start our program
    RenderingPipeline::run(pipeline, config);
}

/**
 * Runs the program as the server side.
 *
 * Server is responsible to accept camera data from client,
 * compute its own GBuffer, and perform raytracing on said buffer to produce
 * visibility bitmap. The bitmap is send back to the client afterwards.
 */
void runServer()
{
    RenderConfig::setConfiguration({ RenderConfig::BufferType::VisibilityBitmap });

    // Define a set of mConfig / window parameters for our program
    SampleConfig config;
    config.windowDesc.title = "NRender UDP Server";
    config.windowDesc.resizableWindow = true;

    // Set up server - configure the sockets and await client connection. We need to await
    // the client connection before we allow the server thread to create the textures, because
    // we want to initialize our server textures the same size as the client
    int texWidth, texHeight;

    ResourceManager::mServerNetworkManager->SetUpServerUdp(DEFAULT_PORT_UDP, texWidth, texHeight);

    config.windowDesc.height = texHeight;
    config.windowDesc.width = texWidth;

    // Create our rendering pipeline
    RenderingPipeline* pipeline = new RenderingPipeline(true, uint2(texWidth, texHeight));

    RenderConfiguration renderConfiguration = getServerRenderConfig(renderMode, renderType, sceneIdx);

    CreatePipeline(renderConfiguration, pipeline);

    // ============================ //
    // Set presets for the pipeline //
    // ============================ //
    if (renderMode == RenderMode::HybridRender) {
        //pipeline->setPresets({
        //    RenderingPipeline::PresetData("Network visibility", "VisibilityBitmap", { 1, 1, 1, 1, 1, 1 })
        //    });
        pipeline->setPresets({
            RenderingPipeline::PresetData("Global Illumination", "Direct / Indirect Illumination", { 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 })
            });
    }
    else if (renderMode == RenderMode::RemoteRender) {
        pipeline->setPresets({
            RenderingPipeline::PresetData("Rendered scene", "", { 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 })
            });
    }
    // Start our program
    RenderingPipeline::run(pipeline, config);
}

/**
 * Runs the program as the client side.
 *
 * Client is responsible to send camera data to the server and
 * waits for visibility bitmap from the server. Client will make use of the received
 * visibility bitmap to compute the final sceneIndex.
 */
void runClient()
{
    RenderConfig::setConfiguration({ RenderConfig::BufferType::VisibilityBitmap });

    // Define a set of mConfig / window parameters for our program
    SampleConfig config;
    config.windowDesc.title = "NRender UDP";
    config.windowDesc.resizableWindow = true;

    // Create our rendering pipeline
    RenderingPipeline* pipeline = new RenderingPipeline(true, uint2(1920, 1080));

    //pipeline->setDefaultSceneName(defaultSceneNames[0]);
    //pipeline->updateEnvironmentMap(environmentMaps[0]);

    // 003 SERVER
    ResourceManager::mClientNetworkManager->SetUpClientUdp("172.26.191.146", DEFAULT_PORT_UDP);
    // 004 SERVER
    //ResourceManager::mClientNetworkManager->SetUpClientUdp("172.26.191.73", DEFAULT_PORT_UDP);
    // 005 SERVER
    //ResourceManager::mClientNetworkManager->SetUpClientUdp("172.26.191.146", DEFAULT_PORT_UDP);

    RenderConfiguration renderConfiguration = getClientRenderConfig(renderMode, renderType, sceneIdx);

    CreatePipeline(renderConfiguration, pipeline);

    // ============================ //
    // Set presets for the pipeline //
    // ============================ //
    if (renderMode == RenderMode::HybridRender) {
        pipeline->setPresets({
            RenderingPipeline::PresetData("Camera Data Transfer GPU-CPU", "V-shading", { 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 })
            });
    }
    else if (renderMode == RenderMode::RemoteRender) {
        pipeline->setPresets({
            RenderingPipeline::PresetData("Camera Data Transfer GPU-CPU", "V-shading", { 1, 1, 1, 1, 1, 1, 1, 1 })
            });
    }
    OutputDebugString(L"\n\n================================PIPELINE CLIENT IS CONFIGURED=================\n\n");

    // Start our program
    RenderingPipeline::run(pipeline, config);
}