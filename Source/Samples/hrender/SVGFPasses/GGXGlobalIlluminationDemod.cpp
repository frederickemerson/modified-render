/**********************************************************************************************************************
# Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without modification, are permitted provided that the
# following conditions are met:
#  * Redistributions of code must retain the copyright notice, this list of conditions and the following disclaimer.
#  * Neither the name of NVIDIA CORPORATION nor the names of its contributors may be used to endorse or promote products
#    derived from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.  IN NO EVENT
# SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA,
# OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
# LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
# ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
**********************************************************************************************************************/

#include "GGXGlobalIlluminationDemod.h"

namespace {
    // Where is our shaders located?
    const char* kFileRayTrace = "Samples\\hrender\\SVGFPasses\\Data\\SVGFPasses\\ggxGlobalIlluminationDemod.rt.hlsl";

    // What are the entry points in that shader for various ray tracing shaders?
    const char* kEntryPointRayGen        = "SimpleDiffuseGIRayGen";

    const char* kEntryPointMiss0         = "ShadowMiss";
    const char* kEntryShadowAnyHit       = "ShadowAnyHit";
    const char* kEntryShadowClosestHit   = "ShadowClosestHit";

    const char* kEntryPointMiss1         = "IndirectMiss";
    const char* kEntryIndirectAnyHit     = "IndirectAnyHit";
    const char* kEntryIndirectClosestHit = "IndirectClosestHit";
};

bool GGXGlobalIlluminationPassDemod::initialize(RenderContext* pRenderContext, ResourceManager::SharedPtr pResManager)
{
    // Stash a copy of our resource manager so we can get rendering resources
    mpResManager = pResManager;

    // Let our resource manager know that we expect some input buffers
    mpResManager->requestTextureResource("WorldPosition");     // Our fragment position, from G-buffer pass
    mpResManager->requestTextureResource("WorldNormal");       // Our fragment normal, from G-buffer pass
    mpResManager->requestTextureResource("__TextureData");     // Compacted material texture data, contains diffuse, specular, emissive colors and other data 
    mpResManager->requestTextureResource(ResourceManager::kEnvironmentMap);  // Our environment map

    // We'll be creating some output buffers.  We store illumination and albedo separately, so we can just
    //     filter illumination without blurring the albdeo (which we know accurately from our G-buffer)
    mpResManager->requestTextureResource(mDirectOutName);      // A buffer to store the direct illumination of each pixel's sample
    mpResManager->requestTextureResource(mIndirectOutName);    // A buffer to store the indirect illumination of each pixel's sample
    mpResManager->requestTextureResource("OutDirectAlbedo");   // A buffer to store the direct albedo of each pixel 
    mpResManager->requestTextureResource("OutIndirectAlbedo"); // A buffer to store the indirect albedo of each pixel

    // Set the default scene to load
    mpResManager->setDefaultSceneName("pink_room/pink_room.fscene");

    // Create our wrapper around a ray tracing pass; specify the entry point for our ray generation shader
    mpRays = RayLaunch::create(kFileRayTrace, kEntryPointRayGen);

    // Add ray type 0 (in this case, our shadow ray)
    mpRays->addMissShader(kFileRayTrace, kEntryPointMiss0);                       
    mpRays->addHitShader(kFileRayTrace, kEntryShadowClosestHit, kEntryShadowAnyHit);

    // Add ray type 1 (in this case, our indirect ray)
    mpRays->addMissShader(kFileRayTrace, "IndirectMiss");                     
    mpRays->addHitShader(kFileRayTrace, "IndirectClosestHit", "IndirectAnyHit");

    // Now that we've passed all our shaders in, compile and (if available) setup the scene
    mpRays->setMaxRecursionDepth(uint32_t(mMaxPossibleRayDepth));
    if (mpScene)
    {
        mpRays->setScene(mpScene);
        mpRays->compileRayProgram();
    }

    return true;
}

void GGXGlobalIlluminationPassDemod::initScene(RenderContext* pRenderContext, Scene::SharedPtr pScene)
{
    // Stash a copy of the scene and pass it to our ray tracer (if initialized)
    mpScene = pScene;
    if (mpScene)
    {
        mpRays->setScene(mpScene);
        mpRays->compileRayProgram();
    }
}

void GGXGlobalIlluminationPassDemod::renderGui(Gui::Window* pPassWindow)
{
    // Add a GUI in our options window allowing selective enabling / disabling of direct or indirect lighting
    int dirty = 0;
    dirty |= (int)pPassWindow->var("Max RayDepth", mUserSpecifiedRayDepth, 0, mMaxPossibleRayDepth, 0.2f);
    dirty |= (int)pPassWindow->checkbox(mDoDirectGI ? "Compute direct illumination" : "Skipping direct illumination",
                                    mDoDirectGI);
    dirty |= (int)pPassWindow->checkbox(mDoIndirectGI ? "Shooting global illumination rays" : "Skipping global illumination",
                                    mDoIndirectGI);
    if (dirty) setRefreshFlag();
}

void GGXGlobalIlluminationPassDemod::execute(RenderContext* pRenderContext)
{
    // Get explicit pointers to the output buffers we're writing into.   (And clear them before returning the pointers.)
    // Compared to the regular GGX GI pass, there are 4 outputs, since we separate direct/indirect lighting and illumination/albedo
    Texture::SharedPtr pDirectDstTex         = mpResManager->getClearedTexture(mDirectOutName, float4(0.0f, 0.0f, 0.0f, 0.0f));
    Texture::SharedPtr pIndirectDstTex       = mpResManager->getClearedTexture(mIndirectOutName, float4(0.0f, 0.0f, 0.0f, 0.0f));
    Texture::SharedPtr pOutAlbedoTex         = mpResManager->getClearedTexture("OutDirectAlbedo", float4(0.0f, 0.0f, 0.0f, 0.0f));
    Texture::SharedPtr pOutIndirectAlbedoTex = mpResManager->getClearedTexture("OutIndirectAlbedo", float4(1.0f, 1.0f, 1.0f, 1.0f));

    // Do we have all the resources we need to render?  If not, return
    if (!pDirectDstTex || !pIndirectDstTex || !mpRays || !mpRays->readyToRender()) return;

    // Set our ray tracing shader variables 
    auto rayVars = mpRays->getRayVars();
    rayVars["RayGenCB"]["gMinT"]         = mpResManager->getMinTDist();
    rayVars["RayGenCB"]["gFrameCount"]   = mFrameCount++;
    rayVars["RayGenCB"]["gDoIndirectGI"] = mDoIndirectGI;
    rayVars["RayGenCB"]["gDoDirectGI"]   = mDoDirectGI;
    rayVars["RayGenCB"]["gMaxDepth"]     = mUserSpecifiedRayDepth;
    rayVars["RayGenCB"]["gEmitMult"]     = 1.0f;
    rayVars["gPos"]         = mpResManager->getTexture("WorldPosition");
    rayVars["gNorm"]        = mpResManager->getTexture("WorldNormal");
    rayVars["gTexData"]     = mpResManager->getTexture("__TextureData");
    rayVars["gDirectOut"]   = pDirectDstTex;
    rayVars["gIndirectOut"] = pIndirectDstTex;
    rayVars["gOutAlbedo"]   = pOutAlbedoTex;
    rayVars["gIndirAlbedo"] = pOutIndirectAlbedoTex;
    rayVars["gEnvMap"] = mpResManager->getTexture(ResourceManager::kEnvironmentMap);

    // Shoot our rays and shade our primary hit points
    mpRays->execute( pRenderContext, mpResManager->getScreenSize() );
}


