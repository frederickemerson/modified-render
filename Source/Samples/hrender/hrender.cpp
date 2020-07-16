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
#include "DxrTutorCommonPasses/AmbientOcclusionPass.h"
#include "DxrTutorCommonPasses/CopyToOutputPass.h"
#include "DxrTutorCommonPasses/GGXGlobalIllumination.h"
#include "DxrTutorCommonPasses/LambertianPlusShadowPass.h"
#include "DxrTutorCommonPasses/LightProbeGBufferPass.h"
#include "DxrTutorCommonPasses/SimpleAccumulationPass.h"
#include "DxrTutorCommonPasses/SimpleDiffuseGIPass.h"
#include "DxrTutorSharedUtils/RenderingPipeline.h"
#include "SVGFPasses/GBufferForSVGF.h"
#include "SVGFPasses/GGXGlobalIlluminationDemod.h"
#include "SVGFPasses/SVGFPass.h"
#include "TestPasses/ModulateAlbedoIllumPass.h"

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
    // -------------------------------- //
    pipeline->setPassOptions(0, {
        // Raytraced GBuffer with camera jitter that allows for depth of field and environment map
        LightProbeGBufferPass::create(),
        // Rasterized GBuffer with support for spatio temporal variance-guided filtering (denoising)
        GBufferForSVGF::create()
    });

    // --------------------------------------- //
    // --- Pass 2 makes use of the GBuffer --- //
    // --------------------------------------- //

    // ------ Pass 2a is an Ambient Occlusion pass
    pipeline->setPass(1, AmbientOcclusionPass::create("Ambient Occlusion"));

    // ------ Pass 2b is a shading pass
    pipeline->setPassOptions(2, {
        // Lambertian BRDF for local lighting, 1 shadow ray per light
        LambertianPlusShadowPass::create("Lambertian Plus Shadows"),
        // Lambertian BRDF for local lighting, 1 shadow ray and 1 scatter ray per pixel
        SimpleDiffuseGIPass::create("Simple Diffuse GI Ray"),
        // GGX BRDF for local lighting, 1 shadow ray and 1 scatter ray (ggx or diffuse) per pixel
        GGXGlobalIlluminationPass::create("Global Illum., GGX BRDF"),
        // GGX BRDF (same as above) with demodulated output. 4 outputs - albedo/illumination X direct/indirect lighting
        GGXGlobalIlluminationPassDemod::create("DirectIllum", "IndirectIllum")
    });

    // ------ Pass 2c is a recombination pass
    pipeline->setPassOptions(3, {
        // This purely re-modulates the output of the demondulated pass
        ModulateAlbedoIllumPass::create("DirectIllum", "IndirectIllum", "Modulate Albedo/Illum"),
        // This performs SVGF denoising, then re-modulates the illumination/albedo 
        SVGFPass::create("DirectIllum", "IndirectIllum", "SVGF Output")
    });

    // --------------------------------------------------------------- //
    // --- Pass 3 just lets us select which pass to view on screen --- //
    // --------------------------------------------------------------- //
    pipeline->setPass(4, CopyToOutputPass::create());

    // ---------------------------------------------------------- //
    // --- Pass 4 temporally accumulates frames for denoising --- //
    // ---------------------------------------------------------- //
    pipeline->setPass(5, SimpleAccumulationPass::create(ResourceManager::kOutputChannel));

    // ============================ //
    // Set presets for the pipeline //
    // ============================ //

    // Presets are "1-indexed", option 0 is the null option to disable the pass
    std::vector<uint32_t> normalGBuff_ggxGI_Options         = { 1, 0, 3, 0, 1, 1 }; 
    std::vector<uint32_t> svgfGBuff_ggxGI_Options           = { 2, 0, 4, 1, 1, 1 };
    std::vector<uint32_t> svgfGBuff_ggxGIDenoised_Options   = { 2, 0, 4, 2, 1, 1 };
    std::vector<uint32_t> normalGBuff_AO_Options            = { 1, 1, 0, 0, 1, 1 };

    pipeline->setPresets({
        RenderingPipeline::PresetData("Global Illum", "Global Illum., GGX BRDF", normalGBuff_ggxGI_Options),
        RenderingPipeline::PresetData("Global Illum (Demodulated GBuffer)", "Modulate Albedo/Illum", svgfGBuff_ggxGI_Options),
        RenderingPipeline::PresetData("Global Illum Denoised", "SVGF Output", svgfGBuff_ggxGIDenoised_Options),
        RenderingPipeline::PresetData("Ambient Occlusion", "Ambient Occlusion", normalGBuff_AO_Options)
    });

    // Start our program
    RenderingPipeline::run(pipeline, config);
    return 0;
}
