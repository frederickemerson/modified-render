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

#include "GGXClientGlobalIllumPass.h"

// Some global vars, used to simplify changing shader location & entry points
namespace {
    // Where is our shaders located?
    const char* kFileRayTrace = "Samples\\hrender\\DxrTutorCommonPasses\\Data\\CommonPasses\\ggxClientGlobalIllumination.rt.hlsl";

    // What are the entry points in that shader for various ray tracing shaders?
    const char* kEntryPointRayGen        = "SimpleDiffuseGIRayGen";
};

bool GGXClientGlobalIllumPass::initialize(RenderContext* pRenderContext, ResourceManager::SharedPtr pResManager)
{
    // Stash a copy of our resource manager so we can get rendering resources
    mpResManager = pResManager;
    mpResManager->requestTextureResources({ "WorldPosition", "WorldNormal", "__TextureData" });
    mpResManager->requestTextureResource("ClientGlobalIllum", ResourceFormat::RGBA8Uint);
    mpResManager->requestTextureResource(mDirectIllumTex, ResourceFormat::RGBA32Float, ResourceManager::kDefaultFlags, mTexWidth, mTexHeight);
    mpResManager->requestTextureResource(mIndirectIllumTex, ResourceFormat::RGBA32Float, ResourceManager::kDefaultFlags, mTexWidth, mTexHeight);
    //mpResManager->requestTextureResource(ResourceManager::kEnvironmentMap);

    // Create our wrapper around a ray tracing pass.  Tell it where our ray generation shader and ray-specific shaders are
    mpRays = RayLaunch::create(0, 1, kFileRayTrace, kEntryPointRayGen);

    if (mpScene)
    {
        mpRays->setScene(mpScene);
        mpRays->compileRayProgram();
    }

    return true;
}

void GGXClientGlobalIllumPass::initScene(RenderContext* pRenderContext, Scene::SharedPtr pScene)
{
    // Stash a copy of the scene and pass it to our ray tracer (if initialized)
    mpScene = pScene;
    if (mpScene)
    {
        mpRays->setScene(mpScene);
        mpRays->compileRayProgram();
    }
}

void GGXClientGlobalIllumPass::renderGui(Gui::Window* pPassWindow)
{
    // Toggling of rays is done in GGXServerGlobalIllumPass instead.
    int dirty = 0;
    if (dirty) setRefreshFlag();
}


void GGXClientGlobalIllumPass::execute(RenderContext* pRenderContext)
{
    // Get the output buffer we're writing into
    Texture::SharedPtr pDirectIllumDstTex = mpResManager->getClearedTexture(mDirectIllumTex, float4(0.0f));
    Texture::SharedPtr pIndirectIllumDstTex = mpResManager->getClearedTexture(mIndirectIllumTex, float4(0.0f));

    // Do we have all the resources we need to render?  If not, return
    if (!pDirectIllumDstTex || !pIndirectIllumDstTex || !mpRays || !mpRays->readyToRender()) return;

    // Set our variables into the global HLSL namespace
    auto globalVars = mpRays->getRayVars();
    globalVars["GlobalCB"]["gFrameCount"]   = mFrameCount++;
    globalVars["GlobalCB"]["gDoIndirectGI"] = mDoIndirectGI;
    globalVars["GlobalCB"]["gDoDirectGI"]   = mDoDirectGI;
    globalVars["GlobalCB"]["gEmitMult"]     = 1.0f;
    globalVars["gPos"]         = mpResManager->getTexture("WorldPosition");
    globalVars["gNorm"]        = mpResManager->getTexture("WorldNormal");
    globalVars["gTexData"]     = mpResManager->getTexture("__TextureData");
    globalVars["gGIData"]      = mpResManager->getTexture("ClientGlobalIllum");
    globalVars["gDirectIllumOut"] = pDirectIllumDstTex;
    globalVars["gIndirectIllumOut"] = pIndirectIllumDstTex;
    globalVars["gEnvMap"]      = mpResManager->getTexture(ResourceManager::kEnvironmentMap);

    // Shoot our rays and shade our primary hit points
    mpRays->execute( pRenderContext, uint2(pDirectIllumDstTex->getWidth(), pDirectIllumDstTex->getHeight()));
}
