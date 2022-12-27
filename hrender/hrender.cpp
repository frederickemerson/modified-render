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
#include "DxrTutorCommonPasses/YUVToRGBAPass.h"
#include "DxrTutorCommonPasses/AmbientOcclusionPass.h"
#include "DxrTutorCommonPasses/GGXClientGlobalIllumPass.h"
#include "DxrTutorCommonPasses/GGXServerGlobalIllumPass.h"
#include "SVGFPasses/SVGFServerPass.h"
#include "SVGFPasses/SVGFClientPass.h"

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

// Switches rendering modes between HybridRender and RemoteRender
RenderMode renderMode = RenderMode::HybridRender;
//RenderMode renderMode = RenderMode::RemoteRender;

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
        else if (renderConfiguration.passOrder[i] == YUVToRGBAPass) {
            // Converts YUV444P to RGBA, Takes in texture name to convert and texture name to output to
            // If output texture is named V-shading, it causes problems during debug mode.
            pipeline->setPass(i, YUVToRGBAPass::create("__V-shadingYUVClient", "V-shading")); 
        }
        else if (renderConfiguration.passOrder[i] == AmbientOcclusionPass) {
            // This pass exists but has been replaced in favour of the GlobalIllum passes.
            pipeline->setPass(i, AmbientOcclusionPass::create("AmbientOcclusion", renderConfiguration.texWidth, renderConfiguration.texHeight));
        }
        else if (renderConfiguration.passOrder[i] == ServerGlobalIllumPass) {
            pipeline->setPass(i, GGXServerGlobalIllumPass::create("OutIndirectColor", "OutIndirectAlbedo", renderConfiguration.texWidth, renderConfiguration.texHeight));
        }
        else if (renderConfiguration.passOrder[i] == ClientGlobalIllumPass) {
            pipeline->setPass(i, GGXClientGlobalIllumPass::create("OutDirectColor", "OutDirectAlbedo", renderConfiguration.texWidth, renderConfiguration.texHeight));
        }
        else if (renderConfiguration.passOrder[i] == SVGFServerPass) {
            pipeline->setPass(i, SVGFServerPass::create("OutIndirectColor", "OutIndirectAlbedo"));
        }
        else if (renderConfiguration.passOrder[i] == SVGFClientPass) {
            pipeline->setPass(i, SVGFClientPass::create("OutDirectColor", "DirectLighting"));
        }
    }
}

/**
 * Runs the program as a stand alone program (no server, no client).
 */
void runDebug()
{
    // hrender config
    RenderConfig::setConfiguration( { RenderConfig::BufferType::VisibilityBitmap } );

    // Define a set of mConfig / window parameters for our program
    SampleConfig config;
    config.windowDesc.title = "NRender";
    config.windowDesc.resizableWindow = true;
    
    RenderConfiguration renderConfiguration;
    if (renderMode == RenderMode::HybridRender) {
        renderConfiguration = getRenderConfigDebugHybrid();
    }
    else if (renderMode == RenderMode::RemoteRender) {
        renderConfiguration = getRenderConfigDebugRemote();
    }

    // Create our rendering pipeline
    RenderingPipeline* pipeline = new RenderingPipeline(true, uint2(renderConfiguration.texWidth, renderConfiguration.texHeight));

    CreatePipeline(renderConfiguration, pipeline);

    // ============================ //
// Set presets for the pipeline //
// ============================ //
    if (renderMode == RenderMode::HybridRender) {
        pipeline->setPresets({
            RenderingPipeline::PresetData("Regular shading", "V-shading", { 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 }),
            RenderingPipeline::PresetData("Preview GBuffer", "DecodedGBuffer", { 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1 }),
            RenderingPipeline::PresetData("No compression, no memory transfer", "V-shading", { 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1 })
            });
    }
    else if (renderMode == RenderMode::RemoteRender) {
        pipeline->setPresets({
            RenderingPipeline::PresetData("Regular shading", "V-shading", { 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1  }),
            RenderingPipeline::PresetData("Preview GBuffer", "DecodedGBuffer", { 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1  }),
            RenderingPipeline::PresetData("No compression, no memory transfer", "V-shading", { 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1 })
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

    RenderConfiguration renderConfiguration;
    if (renderMode == RenderMode::HybridRender) {
        renderConfiguration = getRenderConfigServerHybrid();
    }
    else if (renderMode == RenderMode::RemoteRender) {
        renderConfiguration = getRenderConfigServerRemote();
    }

    CreatePipeline(renderConfiguration, pipeline);

    // ============================ //
    // Set presets for the pipeline //
    // ============================ //
    if (renderMode == RenderMode::HybridRender) {
        //pipeline->setPresets({
        //    RenderingPipeline::PresetData("Network visibility", "VisibilityBitmap", { 1, 1, 1, 1, 1, 1 })
        //    });
        pipeline->setPresets({
            RenderingPipeline::PresetData("Global Illumination", "Direct / Indirect Illumination", { 1, 1, 1, 1, 1, 1, 1 })
            });
    }
    else if (renderMode == RenderMode::RemoteRender) {
        pipeline->setPresets({
            RenderingPipeline::PresetData("Rendered scene", "", { 1, 1, 1, 1, 1, 1, 1, 1 })
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
    
    ResourceManager::mClientNetworkManager->SetUpClientUdp("172.26.191.146", DEFAULT_PORT_UDP);

    RenderConfiguration renderConfiguration;
    if (renderMode == RenderMode::HybridRender) {
        renderConfiguration = getRenderConfigClientHybrid();
    }
    else if (renderMode == RenderMode::RemoteRender) {
        renderConfiguration = getRenderConfigClientRemote();
    }

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

/**
* Functions to get corresponding RenderCongurations
*/
RenderConfiguration getRenderConfigDebugHybrid() {
    // Previous implementation: visibility bitmap + ambient occlusion
    //return {
    //   1920, 1080, // texWidth and texHeight
    //   0, // sceneIndex
    //   12,
    //   { // Array of RenderConfigPass
    //       // --- RenderConfigPass 1 creates a GBuffer --- //
    //       JitteredGBufferPass,
    //       // --- RenderConfigPass 2 makes use of the GBuffer determining visibility under different lights --- //
    //       VisibilityPass,
    //       // --- RenderConfigPass 3 performs per-pixel ambient occlusion --- //
    //       AmbientOcclusionPass,
    //       // --- RenderConfigPass 4 transfers GPU information into CPU --- //
    //       MemoryTransferPassGPU_CPU,
    //       // --- RenderConfigPass 5 compresses buffers to be sent across Network --- //
    //       CompressionPass,
    //       // --- RenderConfigPass 6 simulates delay across network --- //
    //       SimulateDelayPass,
    //       // --- RenderConfigPass 7 decompresses buffers sent across Network--- //
    //       DecompressionPass,
    //       // --- RenderConfigPass 8 transfers CPU information into GPU --- //
    //       MemoryTransferPassCPU_GPU,
    //       // --- RenderConfigPass 9 performs prediction on visibility bitmap if frames are behind. --- //
    //       PredictionPass,
    //       // --- RenderConfigPass 10 makes use of the visibility bitmap to shade the sceneIndex. We also provide the ability to preview the GBuffer alternatively. --- //
    //       VShadingPass,
    //       // --- RenderConfigPass 11 just lets us select which pass to view on screen --- //
    //       CopyToOutputPass,
    //       // --- RenderConfigPass 12 temporally accumulates frames for denoising --- //
    //       SimpleAccumulationPass
    //   }
    //};
    return {
        1920, 1080, // texWidth and texHeight
        0, // sceneIndex
        12,
        { // Array of RenderConfigPass
            // --- JitteredGBufferPass creates a GBuffer --- //
            JitteredGBufferPass,
            // --- VisibilityPass makes use of the GBuffer determining visibility under different lights --- //
            //VisibilityPass,
            // --- ServerGlobalIllumPass computes indirect illumination color and
            //     selects random light index to be used for direct illumination. --- //
            ServerGlobalIllumPass,
            // --- MemoryTransferPassGPU_CPU transfers GPU information into CPU --- //
            MemoryTransferPassGPU_CPU,
            // --- CompressionPass compresses buffers to be sent across Network --- //
            CompressionPass,
            // --- SimulateDelayPass simulates delay across network --- //
            SimulateDelayPass,
            // --- DecompressionPass decompresses buffers sent across Network--- //
            DecompressionPass,
            // --- MemoryTransferPassCPU_GPU transfers CPU information into GPU --- //
            MemoryTransferPassCPU_GPU,
            // --- ClientGlobalIllumPass loads server indirect illumination into texture and
            //     calculates direct illumination using given random light index --- //
            ClientGlobalIllumPass,
            // --- PredictionPass performs prediction on visibility bitmap if frames are behind. --- //
            PredictionPass,
            // --- VShadingPass makes use of the direct and indirect lighting to shade the sceneIndex.
            //     We also provide the ability to preview the GBuffer alternatively. --- //
            VShadingPass,
            // --- CopyToOutputPass just lets us select which pass to view on screen --- //
            CopyToOutputPass,
            // --- SimpleAccumulationPass temporally accumulates frames for denoising --- //
            SimpleAccumulationPass
        }
    };
}

RenderConfiguration getRenderConfigServerHybrid() {
    return {
        1920, 1080, // texWidth and texHeight
        0, // sceneIndex
        7,
        { // Array of RenderConfigPass
            // --- NetworkServerRecvPass receives camera data from client --- //
            NetworkServerRecvPass,
            // --- JitteredGBufferPass creates a GBuffer on server side--- //
            JitteredGBufferPass,
            // --- VisibilityPass makes use of the GBuffer determining visibility under different lights --- //
            //VisibilityPass,
            // --- AmbientOcclusionPass performs per-pixel ambient occlusion --- //
            //AmbientOcclusionPass,
            // --- ServerGlobalIllumPass computes indirect illumination color and
            //     selects random light index to be used for direct illumination. --- //
            ServerGlobalIllumPass,
            // --- SVGFServerPass performs denoising on indirect illumination.
            SVGFServerPass,
            // --- MemoryTransferPassGPU_CPU transfers GPU information into CPU --- //
            MemoryTransferPassGPU_CPU,
            // --- CompressionPass compresses buffers to be sent across network --- //
            CompressionPass,
            // --- NetworkServerSendPass sends the visibility bitmap to the client --- //
            NetworkServerSendPass
        }
    };
}

RenderConfiguration getRenderConfigClientHybrid() {
    // Previous implementation for client.
    //return {
    //    1920, 1080, // texWidth and texHeight
    //    0, // sceneIndex
    //    9,
    //    { // Array of RenderConfigPass
    //        // --- RenderConfigPass 1 Send camera data to server --- //
    //        NetworkClientSendPass,
    //        // --- RenderConfigPass 2 receive visibility bitmap from server --- //
    //        NetworkClientRecvPass,
    //        // --- RenderConfigPass 3 decompresses buffers sent across network--- //
    //        DecompressionPass,
    //        // --- RenderConfigPass 4 transfers CPU information into GPU --- //
    //        MemoryTransferPassCPU_GPU,
    //        // --- RenderConfigPass 5 performs prediction on visibility bitmap if frames are behind. --- //
    //        PredictionPass,
    //        // --- RenderConfigPass 6 makes use of the visibility bitmap to shade the sceneIndex. --- //
    //        VShadingPass,
    //        // --- RenderConfigPass 7 just lets us select which pass to view on screen --- //
    //        CopyToOutputPass,
    //        // --- RenderConfigPass 8 temporally accumulates frames for denoising --- //
    //        SimpleAccumulationPass,
    //        // --- RenderConfigPass 9 creates a GBuffer on client side--- //
    //        JitteredGBufferPass
    //    }
    //};
    return {
        1920, 1080, // texWidth and texHeight
        0, // sceneIndex
        11,
        { // Array of RenderConfigPass
            // --- JitteredGBufferPass creates a GBuffer on client side--- //
            JitteredGBufferPass,
            // --- NetworkClientSendPass sends camera data to server --- //
            NetworkClientSendPass,
            // --- NetworkClientRecvPass receives visibility bitmap from server --- //
            NetworkClientRecvPass,
            // --- DecompressionPass decompresses buffers sent across network--- //
            DecompressionPass,
            // --- MemoryTransferPassCPU_GPU transfers CPU information into GPU --- //
            MemoryTransferPassCPU_GPU,
            // --- ClientGlobalIllumPass loads server indirect illumination into texture and
            //     calculates direct illumination using given random light index --- //
            ClientGlobalIllumPass,
            // --- SVGFClientPass performs denoising on direct illumination
            SVGFClientPass,
            // --- PredictionPass performs prediction on visibility bitmap if frames are behind. --- //
            PredictionPass,
            // --- VShadingPass modulates the direct and indirect illumination. --- //
            VShadingPass,
            // --- CopyToOutputPass lets us select which pass to view on screen --- //
            CopyToOutputPass,
            // --- SimpleAccumulationPass temporally accumulates frames for denoising --- //
            SimpleAccumulationPass,
        }
    };
}

RenderConfiguration getRenderConfigDebugRemote() {
    return {
    1920, 1080, // texWidth and texHeight
    0, // sceneIndex
    12,
    { // Array of RenderConfigPass
        // --- RenderConfigPass 1 creates a GBuffer --- //
        JitteredGBufferPass,
        // --- RenderConfigPass 2 makes use of the GBuffer determining visibility under different lights --- //
        VisibilityPass,
        // --- RenderConfigPass 3 performs prediction on visibility bitmap if frames are behind. --- //
        PredictionPass,
        // --- RenderConfigPass 4 makes use of the visibility bitmap to shade the sceneIndex in YUV format. ---//
        // --- We also provide the ability to preview the GBuffer alternatively. --- //
        VShadingPass,
        // --- RenderConfigPass 5 transfers GPU information into CPU --- //
        MemoryTransferPassGPU_CPU,
        // --- RenderConfigPass 6 compresses buffers to be sent across Network --- //
        CompressionPass,
        // --- RenderConfigPass 7 simulates delay across network --- //
        SimulateDelayPass,
        // --- RenderConfigPass 8 decompresses buffers sent across Network--- //
        DecompressionPass,
        // --- RenderConfigPass 9 transfers CPU information into GPU --- //
        MemoryTransferPassCPU_GPU,
        // --- RenderConfigPass 10 converts the YUV data back to RGBA format --- //
        YUVToRGBAPass,
        // --- RenderConfigPass 11 just lets us select which pass to view on screen --- //
        CopyToOutputPass,
        // --- RenderConfigPass 12 temporally accumulates frames for denoising --- //
        SimpleAccumulationPass
    }
    };
}

RenderConfiguration getRenderConfigServerRemote() {
    return {
    1920, 1080, // texWidth and texHeight
    0, // sceneIndex
    8,
    { // Array of RenderConfigPass
        // --- RenderConfigPass 1 Receive camera data from client --- //
        NetworkServerRecvPass,
        // --- RenderConfigPass 2 creates a GBuffer on server side--- //
        JitteredGBufferPass,
        // --- RenderConfigPass 3 makes use of the GBuffer determining visibility under different lights --- //
        VisibilityPass,
        // --- RenderConfigPass 4 performs prediction on visibility bitmap if frames are behind. --- //
        PredictionPass,
        // --- RenderConfigPass 5 makes use of the visibility bitmap to shade the sceneIndex. --- //
        VShadingPass,
        // --- RenderConfigPass 6 transfers GPU information into GPU --- //
        MemoryTransferPassGPU_CPU,
        // --- RenderConfigPass 7 compresses buffers to be sent across network --- //
        CompressionPass,
        // --- RenderConfigPass 8 sends the rendered scene to the client --- //
        NetworkServerSendPass
    }
    };
}

RenderConfiguration getRenderConfigClientRemote() {
    return {
    1920, 1080, // texWidth and texHeight
    0, // sceneIndex
    8,
    { // Array of RenderConfigPass
        // --- RenderConfigPass 1 Send camera data to server --- //
        NetworkClientSendPass,
        // --- RenderConfigPass 2 receive rendered scene from server --- //
        NetworkClientRecvPass,
        // --- RenderConfigPass 3 decompresses buffers sent across network--- //
        DecompressionPass,
        // --- RenderConfigPass 4 transfers CPU information into GPU --- //
        MemoryTransferPassCPU_GPU,
        // --- RenderConfigPass 5 converts the YUV data back to RGBA format --- //
        YUVToRGBAPass,
        // --- RenderConfigPass 6 just lets us select which pass to view on screen --- //
        CopyToOutputPass,
        // --- RenderConfigPass 7 temporally accumulates frames for denoising --- //
        SimpleAccumulationPass,
        // --- RenderConfigPass 8 creates a GBuffer on client side--- //
        JitteredGBufferPass
    }
    };
}