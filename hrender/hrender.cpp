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
#include "DxrTutorCommonPasses/JitteredGBufferPass.h"
#include "DxrTutorCommonPasses/LambertianPlusShadowPass.h"
#include "DxrTutorCommonPasses/SimpleAccumulationPass.h"
#include "DxrTutorSharedUtils/RenderingPipeline.h"
#include "DxrTutorSharedUtils/NetworkManager.h"
#include "NetworkPasses/VisibilityPass.h"
#include "NetworkPasses/VShadingPass.h"
#include "NetworkPasses/MemoryTransferPassClientCPU_GPU.h"
#include "NetworkPasses/MemoryTransferPassClientGPU_CPU.h"
#include "NetworkPasses/MemoryTransferPassServerCPU_GPU.h"
#include "NetworkPasses/MemoryTransferPassServerGPU_CPU.h"
#include "NetworkPasses/NetworkPass.h"

void runServer();
void runClient();
void runDebug();

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

void runDebug()
{
    // Define a set of config / window parameters for our program
    SampleConfig config;
    config.windowDesc.title = "NRender";
    config.windowDesc.resizableWindow = true;

    // Create our rendering pipeline
    RenderingPipeline* pipeline = new RenderingPipeline();


    pipeline->setPassOptions(0, {
        JitteredGBufferPass::create()
    });

    pipeline->setPassOptions(1, {
        VisibilityPass::create("VisibilityBitmap", "WorldPosition"),
    });
    pipeline->setPassOptions(2, {
        VShadingPass::create("V-shading"),
    });
    pipeline->setPass(3, CopyToOutputPass::create());
    // ---------------------------------------------------------- //
    // --- Pass 7 temporally accumulates frames for denoising --- //
    pipeline->setPass(4, SimpleAccumulationPass::create(ResourceManager::kOutputChannel));

    // ============================ //
    // Set presets for the pipeline //
    // ============================ //
    pipeline->setPresets({
        RenderingPipeline::PresetData("Regular shading", "V-shading", { 1, 1, 1, 1, 1 })
        });

    OutputDebugString(L"\n\n================================PIPELINE CLIENT IS CONFIGURED=================");

    // Start our program
    RenderingPipeline::run(pipeline, config);
}

void runServer()
{
    // Define a set of config / window parameters for our program
    SampleConfig config;
    config.windowDesc.title = "NRender Server";
    config.windowDesc.resizableWindow = true;

    // Set up server - configure the sockets and await client connection. We need to await
    // the client connection before we allow the server thread to create the textures, because
    // we want to initialize our server textures the same size as the client
    int texWidth, texHeight;
    ResourceManager::mNetworkManager->SetUpServer(DEFAULT_PORT, texWidth, texHeight);
    NetworkPass::posTexHeight = texHeight;
    NetworkPass::posTexWidth = texWidth;
    config.windowDesc.height = texHeight;
    config.windowDesc.width = texWidth;

    // Create our rendering pipeline
    RenderingPipeline* pipeline = new RenderingPipeline(true, uint2(texWidth, texHeight));
    // ---------------------------------------------- //
    // --- Pass 1 Receive camera data from client --- //
    pipeline->setPassOptions(0, {
        NetworkPass::create(NetworkPass::Mode::Server, texWidth, texHeight),
    });
    // ---------------------------------------------- //
    // --- Pass 2 creates a GBuffer on server side--- //
    pipeline->setPassOptions(1, {
        // Rasterized GBuffer 
        JitteredGBufferPass::create(texWidth, texHeight)
    });
    // ------------------------------------------------------------------------------------- //
    // --- Pass 3 makes use of the GBuffer determining visibility under different lights --- //
    pipeline->setPassOptions(2, {
        // Lambertian BRDF for local lighting, 1 shadow ray per light
        VisibilityPass::create("VisibilityBitmap", "WorldPosition", texWidth, texHeight)
    });
    // ------------------------------------------------- //
    // --- Pass 4 transfers GPU information into CPU --- //
    pipeline->setPassOptions(3, {
        // Memory transfer from GPU to CPU
        MemoryTransferPassServerGPU_CPU::create()
    });
    // ---------------------------------------------------- //
    // --- Pass 5 Send visibility bitmap back to client --- //
    pipeline->setPassOptions(4, {
        NetworkPass::create(NetworkPass::Mode::ServerSend, texWidth, texHeight),
    });
    pipeline->setPassOptions(5, {
        // Make use of the received visibility bitmap to construct final scene
        VShadingPass::create("V-shading"),
        });
    // --------------------------------------------------------------- //
    // --- Pass 6 just lets us select which pass to view on screen --- //
    pipeline->setPass(6, CopyToOutputPass::create());
    // ---------------------------------------------------------- //
    // --- Pass 7 temporally accumulates frames for denoising --- //
    pipeline->setPass(7, SimpleAccumulationPass::create(ResourceManager::kOutputChannel));

    // ============================ //
    // Set presets for the pipeline //
    // ============================ //
    pipeline->setPresets({
        RenderingPipeline::PresetData("Network visibility", "VisibilityBitmap", { 1, 1, 1, 1, 1, 1, 1, 1 })
    });

    OutputDebugString(L"\n\n================================PIPELINE RENDER SERVER IS CONFIGURED=================\n\n");


    // Start our program
    RenderingPipeline::run(pipeline, config);
}

void runClient()
{
    // Define a set of config / window parameters for our program
    SampleConfig config;
    config.windowDesc.title = "NRender";
    config.windowDesc.resizableWindow = true;

    // Create our rendering pipeline
    RenderingPipeline* pipeline = new RenderingPipeline();
    
    ResourceManager::mNetworkManager->SetUpClient("192.168.1.111", DEFAULT_PORT);
    
    // ---------------------------------------- //
    // --- Pass 1 Send camera data to server--- //
    pipeline->setPassOptions(0, {
        // Send scene and camera data across network to server, and re-receive the visibility bitmap
        NetworkPass::create(NetworkPass::Mode::ClientSend)
    });
    // ---------------------------------------------- //
    // --- Pass 2 creates a GBuffer on client side--- //
    pipeline->setPassOptions(1, {
        // Rasterized GBuffer 
        JitteredGBufferPass::create()
    });
    // ---------------------------------------------------- //
    // --- Pass 3 receive visibility bitmap from server --- //
    pipeline->setPassOptions(2, {
        // Re-receive the visibility bitmap
        NetworkPass::create(NetworkPass::Mode::Client)
    });
    // ------------------------------------------------- //
    // --- Pass 4 transfers CPU information into GPU --- //
    pipeline->setPassOptions(3, {
        // Memory transfer from CPU to GPU
        MemoryTransferPassClientCPU_GPU::create()
        });
    // -------------------------------------------------------------------- //
    // --- Pass 5 makes use of the visibility bitmap to shade the scene --- //
    pipeline->setPassOptions(4, {
        // Make use of the received visibility bitmap to construct final scene
        VShadingPass::create("V-shading"),
    });
    // --------------------------------------------------------------- //
    // --- Pass 6 just lets us select which pass to view on screen --- //
    pipeline->setPass(5, CopyToOutputPass::create());
    // ---------------------------------------------------------- //
    // --- Pass 7 temporally accumulates frames for denoising --- //
    pipeline->setPass(6, SimpleAccumulationPass::create(ResourceManager::kOutputChannel));

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
