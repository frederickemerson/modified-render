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
#include "NetworkPasses/NetworkPass.h"

void runServer();
void runClient();

int WINAPI WinMain(_In_ HINSTANCE hInstance, _In_opt_ HINSTANCE hPrevInstance, _In_ LPSTR lpCmdLine, _In_ int nShowCmd)
{
    //NetworkPass::Mode mode = NetworkPass::Mode::Client;
    NetworkPass::Mode mode = NetworkPass::Mode::Server;

    if (mode == NetworkPass::Mode::Client)
        runClient();
    else 
        runServer();
   
    return 0;
}

void runServer()
{
    // Define a set of config / window parameters for our program
    SampleConfig config;
    config.windowDesc.title = "NRender Server";
    config.windowDesc.resizableWindow = true;

    // Create our rendering pipeline
    RenderingPipeline* pipeline = new RenderingPipeline();

    ResourceManager::mNetworkManager->SetUpServer(DEFAULT_PORT);
    ResourceManager::mNetworkManager->AcceptAndListenServer();

    // -------------------------------- //
    // --- Pass 1 creates a GBuffer --- //
    pipeline->setPassOptions(0, {
        NetworkPass::create("Receiver", NetworkPass::Mode::Server),
    });
    // ------------------------------------------------------------------------------------- //
    // --- Pass 2 makes use of the GBuffer determining visibility under different lights --- //
    pipeline->setPassOptions(1, {
        // Lambertian BRDF for local lighting, 1 shadow ray per light
        VisibilityPass::create("VisibilityBitmap"),
        LambertianPlusShadowPass::create("RTLambertian")
    });
    // -------------------------------------------------------------------- //
    // --- Pass 3 makes use of the visibility buffer to shade the scene --- //
    pipeline->setPassOptions(2, {
        NetworkPass::create("Sender", NetworkPass::Mode::ServerSend),
    });

    // --------------------------------------------------------------- //
    // --- Pass 4 just lets us select which pass to view on screen --- //
    pipeline->setPass(3, CopyToOutputPass::create());
    // ---------------------------------------------------------- //
    // --- Pass 5 temporally accumulates frames for denoising --- //
    pipeline->setPass(4, SimpleAccumulationPass::create(ResourceManager::kOutputChannel));

    // ============================ //
    // Set presets for the pipeline //
    // ============================ //
    pipeline->setPresets({
        RenderingPipeline::PresetData("Network visibility", "VisibilityBitmap", { 1, 1, 1, 1, 1 }),
        RenderingPipeline::PresetData("Raytraced Lighting", "RTLambertian", { 1, 2, 0, 1, 1 })
    });

    OutputDebugString(L"\n\n\n\n\n================================PIPELINE SERVER IS CONFIGURED=================\n\n\n\n");


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
    
    ResourceManager::mNetworkManager->SetUpClient("172.25.107.160", DEFAULT_PORT);
    //NetworkManager::SharedPtr networkManager = NetworkManager::create();
    //networkManager->SetUpClient("localhost", "10871");
    //
    //const char* sendbuf = "this is a test";
    //networkManager->SendDataFromClient(sendbuf, (int)strlen(sendbuf), 0);

    // -------------------------------- //
    // --- Pass 1 creates a GBuffer --- //
    pipeline->setPassOptions(0, {
        // Rasterized GBuffer 
        JitteredGBufferPass::create()
    });
    // ------------------------------------------------------------------------------------- //
    // --- Pass 2 makes use of the GBuffer determining visibility under different lights --- //
    pipeline->setPassOptions(1, {
        // Send scene and gbuffer across network to server, and re-receive the visibility bitmap
        NetworkPass::create("Client", NetworkPass::Mode::Client),
        // Lambertian BRDF for local lighting, 1 shadow ray per light
        VisibilityPass::create("VisibilityBitmap"),
        LambertianPlusShadowPass::create("RTLambertian")
    });
    // -------------------------------------------------------------------- //
    // --- Pass 3 makes use of the visibility buffer to shade the scene --- //
    pipeline->setPassOptions(2, {
        //// Lambertian BRDF for local lighting, based on the visibility buffer created in pass 2
        //NetworkPass::create("ServerRecv", NetworkPass::Mode::Server),

        
        VShadingPass::create("V-shading"),
        //LambertianPlusShadowPass::create("RTLambertian"),
        //VisibilityPass::create("VisibilityBitmap")

    });
    //pipeline->setPassOptions(3, {
    //    VisibilityPass::create("VisibilityBitmap")
    //});
    //pipeline->setPassOptions(4, {
    //    NetworkPass::create("ServerSend", NetworkPass::Mode::ServerSend)

    //});
    //pipeline->setPassOptions(5, {
    //    // Lambertian BRDF for local lighting, based on the visibility buffer created in pass 2
    //    VShadingPass::create("V-shading")
    //    });
    // --------------------------------------------------------------- //
    // --- Pass 4 just lets us select which pass to view on screen --- //
    pipeline->setPass(3, CopyToOutputPass::create());
    // ---------------------------------------------------------- //
    // --- Pass 5 temporally accumulates frames for denoising --- //
    pipeline->setPass(4, SimpleAccumulationPass::create(ResourceManager::kOutputChannel));

    // ============================ //
    // Set presets for the pipeline //
    // ============================ //
    pipeline->setPresets({
        //RenderingPipeline::PresetData("CPU transfer, Visibility then Combination", "V-shading", { 1, 1, 1, 1, 1, 1, 1, 1 });
        RenderingPipeline::PresetData("CPU transfer, Visibility then Combination", "V-shading", { 1, 1, 1, 1, 1 })
        //RenderingPipeline::PresetData("Split Visibility then Combination on Client", "V-shading", { 1, 2, 2, 0, 0, 0, 1, 1 }),
        //RenderingPipeline::PresetData("Raytraced Lighting", "RTLambertian", { 1, 3, 0, 0, 0, 0, 1, 1 })
    });

    OutputDebugString(L"\n\n\n\n\n================================PIPELINE CLIENT IS CONFIGURED=================\n\n\n");

    // Start our program
    RenderingPipeline::run(pipeline, config);
}
