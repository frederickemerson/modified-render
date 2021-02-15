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

#include "VisibilityPass.h"

namespace {
    // Where is our shaders located?
    const char* kFileRayTrace = "Samples\\hrender\\NetworkPasses\\Data\\NetworkPasses\\visibilityPass.rt.hlsl";

    // What are the entry points in that shader for various ray tracing shaders?
    const char* kEntryPointRayGen  = "SimpleShadowsRayGen";
    const char* kEntryPointMiss0   = "ShadowMiss";
    const char* kEntryAoAnyHit     = "ShadowAnyHit";
    const char* kEntryAoClosestHit = "ShadowClosestHit";
};

bool VisibilityPass::initialize(RenderContext* pRenderContext, ResourceManager::SharedPtr pResManager)
{
    // Stash a copy of our resource manager so we can get rendering resources
    mpResManager = pResManager;

    // Our GUI needs less space than other passes, so shrink the GUI window.
    setGuiSize(int2(300, 70));

    // Note that we some buffers from the G-buffer, plus the standard output buffer
    mpResManager->requestTextureResource("WorldPosition");
    mpResManager->requestTextureResource("WorldNormal");
    mpResManager->requestTextureResource("__TextureData");
    mOutputIndex = mpResManager->requestTextureResource(mOutputTexName, ResourceFormat::R32Uint);

    // Create our wrapper around a ray tracing pass.  Tell it where our ray generation shader and ray-specific shaders are
    mpRays = RayLaunch::create(kFileRayTrace, kEntryPointRayGen);
    mpRays->addMissShader(kFileRayTrace, kEntryPointMiss0);
    mpRays->addHitShader(kFileRayTrace, kEntryAoClosestHit, kEntryAoAnyHit);

    // Now that we've passed all our shaders in, compile and (if available) setup the scene
    if (mpScene) {
        mpRays->setScene(mpScene);
        mpRays->compileRayProgram();
    }

    return true;
}

void VisibilityPass::initScene(RenderContext* pRenderContext, Scene::SharedPtr pScene)
{
    // Stash a copy of the scene and pass it to our ray tracer (if initialized)
    mpScene = pScene;
    if (!mpScene) return;
    if (mpRays) {
        mpRays->setScene(mpScene);
        mpRays->compileRayProgram();
    }
}

void VisibilityPass::execute(RenderContext* pRenderContext)
{
    // Get the output buffer we're writing into
    Texture::SharedPtr pDstTex = mpResManager->getClearedTexture(mOutputIndex, float4(0.0f));

    // Do we have all the resources we need to render?  If not, return
    if (!pDstTex || !mpRays || !mpRays->readyToRender()) return;

    // Set our ray tracing shader variables 
    auto rayVars = mpRays->getRayVars();
    rayVars["RayGenCB"]["gMinT"] = mpResManager->getMinTDist();
    rayVars["RayGenCB"]["gSkipShadows"] = mSkipShadows;
    rayVars["gPos"]         = mpResManager->getTexture("WorldPosition");
    rayVars["gNorm"]        = mpResManager->getTexture("WorldNormal");
    rayVars["gTexData"]     = mpResManager->getTexture("__TextureData");
    rayVars["gOutput"]      = pDstTex;

    // Shoot our rays and shade our primary hit points
    mpRays->execute( pRenderContext, uint2(pDstTex->getWidth(), pDstTex->getHeight()) );
}

void VisibilityPass::renderGui(Gui::Window* pPassWindow)
{
    int dirty = 0;

    dirty |= (int)pPassWindow->checkbox("Skip shadow computation", mSkipShadows, false);

    // If any of our UI parameters changed, let the pipeline know we're doing something different next frame
    if (dirty) setRefreshFlag();
}

