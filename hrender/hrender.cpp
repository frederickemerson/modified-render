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
#include "DxrTutorCommonPasses/LightProbeGBufferPass.h"
#include "DxrTutorCommonPasses/SimpleAccumulationPass.h"
#include "DxrTutorSharedUtils/RenderingPipeline.h"
#include "RasterizedPasses/RasterLightingPass.h"

constexpr int CLIENT = 0;
constexpr int SERVER = 1;

int WINAPI WinMain(_In_ HINSTANCE hInstance, _In_opt_ HINSTANCE hPrevInstance, _In_ LPSTR lpCmdLine, _In_ int nShowCmd)
{
    // Define a set of config / window parameters for our program
    SampleConfig config;
    config.windowDesc.title = "HRender";
    config.windowDesc.resizableWindow = true;

    // Create our rendering pipeline
    RenderingPipeline* pipeline = new RenderingPipeline();

    // ========================== //
    // Add Passes to the Pipeline //
    // ========================== //

    // -------------------------------- //
    // --- Pass 1 creates a GBuffer --- //
    pipeline->setPassOptions(0, {
        // Rasterized GBuffer 
        JitteredGBufferPass::create(),
        // Raycasted GBuffer with camera jitter that allows for depth of field
        LightProbeGBufferPass::create()
    });
    // --------------------------------------------------- //
    // --- Pass 2 makes use of the GBuffer for shading --- //
    pipeline->setPassOptions(1, {
        // Lambertian BRDF for local lighting, shadow mapping
        RasterLightingPass::create("Rasterized Lighting"),
        // Lambertian BRDF for local lighting, 1 shadow ray per light
        LambertianPlusShadowPass::create("Lambertian Plus Shadows")
    });
    // --------------------------------------------------------------- //
    // --- Pass 3 just lets us select which pass to view on screen --- //
    pipeline->setPass(2, CopyToOutputPass::create());
    // ---------------------------------------------------------- //
    // --- Pass 4 temporally accumulates frames for denoising --- //
    pipeline->setPass(3, SimpleAccumulationPass::create(ResourceManager::kOutputChannel));

    // ============================ //
    // Set presets for the pipeline //
    // ============================ //

    // Presets are "1-indexed", option 0 is the null option to disable the pass
    //std::vector<uint32_t> normalGBuff_rasterized_Options    = ;
    //std::vector<uint32_t> normalGBuff_lambertian_Options    = ;
    //std::vector<uint32_t> lpGBuff_rasterized_Options        = ;
    //std::vector<uint32_t> lpGBuff_lambertian_Options        = ;

    pipeline->setPresets({
        RenderingPipeline::PresetData("Rasterized Lighting (raster gbuffer)", "Rasterized Lighting", { 1, 1, 1, 1 }),
        RenderingPipeline::PresetData("Lambertian Lighting (raster gbuffer)", "Lambertian Plus Shadows", { 1, 2, 1, 1 }),
        RenderingPipeline::PresetData("Rasterized Lighting (raycast gbuffer)", "Rasterized Lighting", { 2, 1, 1, 1 }),
        RenderingPipeline::PresetData("Lambertian Lighting (raycast gbuffer)", "Lambertian Plus Shadows", { 2, 2, 1, 1 })
    });

    // Start our program
    RenderingPipeline::run(pipeline, config);
    return 0;
}
