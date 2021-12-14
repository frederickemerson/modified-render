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
#include "DxrTutorSharedUtils/NetworkManager.h"
#include "NetworkPasses/VisibilityPass.h"
#include "NetworkPasses/VShadingPass.h"
#include "NetworkPasses/MemoryTransferPassClientCPU_GPU.h"
#include "NetworkPasses/MemoryTransferPassServerGPU_CPU.h"
#include "NetworkPasses/NetworkPass.h"

void runServer(bool useTcp);
void runClient(bool useTcp);
void runDebug();

// ========= //
// PINK ROOM //
// ========= //
const char* environmentMap = "MonValley_G_DirtRoad_3k.hdr";
std::string defaultSceneName = "pink_room\\pink_room.pyscene";

// ========== //
// SUN TEMPLE //
// ========== //
//const char* environmentMap = "SunTemple\\SunTemple_Skybox.hdr";
//std::string defaultSceneName = "SunTemple\\SunTemple_v3.pyscene";

// ======= //
// BISTRO //
// ====== //
//const char* environmentMap = "Bistro\\san_giuseppe_bridge_4k.hdr";
//std::string defaultSceneName = "Bistro\\BistroInterior_v4.pyscene";
//std::string defaultSceneName = "Bistro\\BistroExterior_v4.pyscene";

// ======== //
// ZERO DAY //
// ======== //
//const char* environmentMap = "MonValley_G_DirtRoad_3k.hdr";
//std::string defaultSceneName = "ZeroDay\\MEASURE_ONE\\MEASURE_ONE.fbx";

/**
 * Determines the mode or configuration that the program runs
 * based on the command line argument.
 */
int WINAPI WinMain(_In_ HINSTANCE hInstance, _In_opt_ HINSTANCE hPrevInstance, _In_ LPSTR lpCmdLine, _In_ int nShowCmd)
{
    if (std::string(lpCmdLine).find(std::string("no-compression")) != std::string::npos)
    {
        OutputDebugString(L"\n\n======== WITHOUT COMPRESSION =========");
        NetworkManager::mCompression = false;
    }

    if (std::string(lpCmdLine).find(std::string("server")) != std::string::npos)
    {
        OutputDebugString(L"\n\n======== SERVER MODE =========");
        runServer(!(std::string(lpCmdLine).find(std::string("udp")) != std::string::npos));
    }
    else if (std::string(lpCmdLine).find(std::string("client")) != std::string::npos)
    {
        OutputDebugString(L"\n\n======== CLIENT MODE =========");
        runClient(!(std::string(lpCmdLine).find(std::string("udp")) != std::string::npos));
    }
    else
    {
        OutputDebugString(L"\n\n======== DEBUG MODE =========");
        runDebug();
    }
    return 0;
}

/**
 * Runs the program as a stand alone program (no server, no client).
 */
void runDebug()
{
    // Define a set of config / window parameters for our program
    SampleConfig config;
    config.windowDesc.title = "NRender";
    config.windowDesc.resizableWindow = true;

    // Create our rendering pipeline
    RenderingPipeline* pipeline = new RenderingPipeline();
    pipeline->setDefaultSceneName(defaultSceneName);
    pipeline->updateEnvironmentMap(environmentMap);

    // --- Pass 1 creates a GBuffer --- //
    pipeline->setPass(0, JitteredGBufferPass::create());

    // --- Pass 2 makes use of the GBuffer determining visibility under different lights --- //
    pipeline->setPass(1, VisibilityPass::create("VisibilityBitmap", "WorldPosition"));

    // --- Pass 3 makes use of the visibility bitmap to shade the scene. We also provide the ability to preview the GBuffer alternatively. --- //
    pipeline->setPassOptions(2, { VShadingPass::create("V-shading"), DecodeGBufferPass::create("DecodedGBuffer") });

    // --- Pass 4 just lets us select which pass to view on screen --- //
    pipeline->setPass(3, CopyToOutputPass::create());

    // --- Pass 5 temporally accumulates frames for denoising --- //
    pipeline->setPass(4, SimpleAccumulationPass::create(ResourceManager::kOutputChannel));

    // ============================ //
    // Set presets for the pipeline //
    // ============================ //
    pipeline->setPresets({
        RenderingPipeline::PresetData("Regular shading", "V-shading", { 1, 1, 1, 1, 1 }),
        RenderingPipeline::PresetData("Preview GBuffer", "DecodedGBuffer", { 1, 1, 2, 1, 1 })
        });

    // Start our program
    RenderingPipeline::run(pipeline, config);
}

/**
 * Runs the program as the server side.
 *
 * Server is responsible to accept camera data from client,
 * compute its own GBuffer, and perform raytracing on said bufferto produce
 * visibility bitmap. The bitmap is send back to the client afterwards.
 */
void runServer(bool useTcp)
{
    // Define a set of config / window parameters for our program
    SampleConfig config;
    config.windowDesc.title = useTcp ? "NRender Server" : "NRender UDP Server";
    config.windowDesc.resizableWindow = true;

    // Set up server - configure the sockets and await client connection. We need to await
    // the client connection before we allow the server thread to create the textures, because
    // we want to initialize our server textures the same size as the client
    int texWidth, texHeight;

    ResourceManager::mNetworkManager->SetUpServerUdp(DEFAULT_PORT_UDP, texWidth, texHeight);
    
    NetworkPass::posTexHeight = texHeight;
    NetworkPass::posTexWidth = texWidth;
    config.windowDesc.height = texHeight;
    config.windowDesc.width = texWidth;

    // Create our rendering pipeline
    RenderingPipeline* pipeline = new RenderingPipeline(true, uint2(texWidth, texHeight));
    pipeline->setDefaultSceneName(defaultSceneName);
    pipeline->updateEnvironmentMap(environmentMap);

    // --- Pass 1 Receive camera data from client --- //
    pipeline->setPass(0, NetworkPass::create(useTcp ? NetworkPass::Mode::Server : NetworkPass::Mode::ServerUdp, texWidth, texHeight));

    // --- Pass 2 creates a GBuffer on server side--- //
    pipeline->setPass(1, JitteredGBufferPass::create(texWidth, texHeight));

    // --- Pass 3 makes use of the GBuffer determining visibility under different lights --- //
    pipeline->setPass(2, VisibilityPass::create("VisibilityBitmap", "WorldPosition", texWidth, texHeight)); // Lambertian BRDF for local lighting, 1 shadow ray per light

    // --- Pass 4 transfers GPU information into CPU --- //
    pipeline->setPass(3, MemoryTransferPassServerGPU_CPU::create());

    // --- Pass 5 Send visibility bitmap back to client --- //
    pipeline->setPass(4, NetworkPass::create(useTcp ? NetworkPass::Mode::ServerSend : NetworkPass::Mode::ServerUdpSend, texWidth, texHeight));

    // ============================ //
    // Set presets for the pipeline //
    // ============================ //
    pipeline->setPresets({
        RenderingPipeline::PresetData("Network visibility", "VisibilityBitmap", { 1, 1, 1, 1, 1 })
        });

    // Start our program
    RenderingPipeline::run(pipeline, config);
}

/**
 * Runs the program as the client side.
 *
 * Client is responsible to send camera data to the server and
 * waits for visibility bitmap from the server. Client will make use of the received
 * visibility bitmap to compute the final scene.
 */
void runClient(bool useTcp)
{
    // Define a set of config / window parameters for our program
    SampleConfig config;
    config.windowDesc.title = useTcp ? "NRender" : "NRender UDP";
    config.windowDesc.resizableWindow = true;

    // Create our rendering pipeline
    RenderingPipeline* pipeline = new RenderingPipeline();
    pipeline->setDefaultSceneName(defaultSceneName);
    pipeline->updateEnvironmentMap(environmentMap);
    
    ResourceManager::mNetworkManager->SetUpClientUdp("172.26.186.144", DEFAULT_PORT_UDP);

    // --- Pass 1 Send camera data to server--- //
    /*pipeline->setPassOptions(2, {
        NetworkPass::create(NetworkPass::Mode::ClientUdpSendFirst)
    });*/

    // --- Pass 1 Send camera data to server--- //
    pipeline->setPass(0, NetworkPass::create(useTcp ? NetworkPass::Mode::ClientSend : NetworkPass::Mode::ClientUdpSend));

    // --- Pass 2 receive visibility bitmap from server --- //
    pipeline->setPass(1, NetworkPass::create(useTcp ? NetworkPass::Mode::Client : NetworkPass::Mode::ClientUdp));

    // --- Pass 3 transfers CPU information into GPU --- //
    pipeline->setPass(2, MemoryTransferPassClientCPU_GPU::create());

    // --- Pass 4 makes use of the visibility bitmap to shade the scene --- //
    pipeline->setPass(3, VShadingPass::create("V-shading"));

    // --- Pass 5 just lets us select which pass to view on screen --- //
    pipeline->setPass(4, CopyToOutputPass::create());

    // --- Pass 6 temporally accumulates frames for denoising --- //
    pipeline->setPass(5, SimpleAccumulationPass::create(ResourceManager::kOutputChannel));

    // --- Pass 7 creates a GBuffer on client side--- //
    pipeline->setPass(6, JitteredGBufferPass::create()); // Rasterized GBuffer

    // ============================ //
    // Set presets for the pipeline //
    // ============================ //
    pipeline->setPresets({
        RenderingPipeline::PresetData("Camera Data Transfer GPU-CPU", "V-shading", { 1, 1, 1, 1, 1, 1, 1 })
    });

    OutputDebugString(L"\n\n================================PIPELINE CLIENT IS CONFIGURED=================\n\n");

    // Start our program
    RenderingPipeline::run(pipeline, config);
}
