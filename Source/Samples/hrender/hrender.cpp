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
#include "DxrTutorCommonPasses/JitteredGBufferPass.h"
#include "DxrTutorCommonPasses/LambertianPlusShadowPass.h"
#include "DxrTutorCommonPasses/LightProbeGBufferPass.h"
#include "DxrTutorCommonPasses/SimpleAccumulationPass.h"
#include "DxrTutorCommonPasses/SimpleDiffuseGIPass.h"
#include "DxrTutorCommonPasses/SimpleGBufferPass.h"
#include "DxrTutorCommonPasses/ThinLensGBufferPass.h"
#include "DxrTutorSharedUtils/RenderingPipeline.h"

int WINAPI WinMain(_In_ HINSTANCE hInstance, _In_opt_ HINSTANCE hPrevInstance, _In_ LPSTR lpCmdLine, _In_ int nShowCmd)
{
    // Define a set of config / window parameters for our program
    SampleConfig config;
    config.windowDesc.title = "HRender";
    config.windowDesc.resizableWindow = true;

    // Create our rendering pipeline, and add passes to the pipeline
    RenderingPipeline* pipeline = new RenderingPipeline();

    // -------------------------------- //
    // --- Pass 1 creates a GBuffer --- //
    // -------------------------------- //
    pipeline->setPassOptions(0, {
        // Rasterized GBuffer 
        SimpleGBufferPass::create(),
        // Rasterized GBuffer with camera jitter (for antialiasing)
        JitteredGBufferPass::create(),
        // Raytraced GBuffer with camera jitter that allows for depth of field
        ThinLensGBufferPass::create(),
        // Raytraced GBuffer with camera jitter that allows for depth of field and environment map
        LightProbeGBufferPass::create()
    });

    // --------------------------------------- //
    // --- Pass 2 makes use of the GBuffer --- //
    // --------------------------------------- //

    // ------ Pass 2a is a shading pass
    pipeline->setPassOptions(1, {
        // Lambertian BRDF for local lighting, 1 shadow ray per light
        LambertianPlusShadowPass::create("LambertianPlusShadowPass"),
        // Lambertian BRDF for local lighting, 1 shadow ray and 1 scatter ray per point
        SimpleDiffuseGIPass::create("DiffuseGIPass"),
        // GGX BRDF for local lighting, 1 shadow ray and 1 scatter ray (ggx or diffuse) per point
        GGXGlobalIlluminationPass::create("GGXPass")
    });

    // ------ Pass 2b is an Ambient Occlusion pass
    pipeline->setPass(2, AmbientOcclusionPass::create("AmbientOcclusionPass"));

    // --------------------------------------------------------------- //
    // --- Pass 3 just lets us select which pass to view on screen --- //
    // --------------------------------------------------------------- //
    pipeline->setPass(3, CopyToOutputPass::create());

    // ---------------------------------------------------------- //
    // --- Pass 4 temporally accumulates frames for denoising --- //
    // ---------------------------------------------------------- //
    pipeline->setPass(4, SimpleAccumulationPass::create(ResourceManager::kOutputChannel));

    // Start our program
    RenderingPipeline::run(pipeline, config);
    return 0;
}
