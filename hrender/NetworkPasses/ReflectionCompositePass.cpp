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

#include "ReflectionCompositePass.h"

namespace {
    // Where is our shaders located?
    const char* kRCShader = "Samples\\hrender\\NetworkPasses\\Data\\NetworkPasses\\ReflectionCompositePass.ps.hlsl";
};

bool ReflectionCompositePass::initialize(RenderContext* pRenderContext, ResourceManager::SharedPtr pResManager)
{
    // Stash a copy of our resource manager so we can get rendering resources
    mpResManager = pResManager;

    // Our GUI needs less space than other passes, so shrink the GUI window.
    setGuiSize(int2(300, 70));

    // Note that we some buffers from the G-buffer, plus the standard output buffer
    //mpResManager->requestTextureResource("SSRColor", ResourceFormat::RGBA32Float, ResourceManager::kDefaultFlags, mTexWidth, mTexHeight);
    //mpResManager->requestTextureResource("SRTReflection", ResourceFormat::RGBA32Float, ResourceManager::kDefaultFlags, mTexWidth, mTexHeight);
    mpResManager->requestTextureResource("RCColor", ResourceFormat::RGBA32Float, ResourceManager::kDefaultFlags, mTexWidth, mTexHeight);


    // Create our graphics state and shaders
    mpGfxState = GraphicsState::create();
    mpRCShader = FullscreenLaunch::create(kRCShader);

    return true;
}

void ReflectionCompositePass::initScene(RenderContext* pRenderContext, Scene::SharedPtr pScene)
{
    // Stash a copy of the scene and pass it to our ray tracer (if initialized)
    if (pScene) mpScene = pScene;

    mpRCFbo = mpResManager->createManagedFbo({"RCColor" }, "Z-Buffer");
}

void ReflectionCompositePass::execute(RenderContext* pRenderContext)
{
    pRenderContext->clearFbo(mpRCFbo.get(), float4(0.0f), 1, 0);

    Camera::SharedPtr cam = mpScene->getCamera();

    // Set our SSR shader variables
    auto RCVars = mpRCShader->getVars();
    RCVars["RCCB"]["gSkipRC"] = mSkipRC;
    RCVars["RCCB"]["gCamPos"] = cam->getPosition();
    RCVars["gVshading"] = mpResManager->getTexture("V-shading");
    RCVars["gSSRColor"] = mpResManager->getTexture("SSRColor");
    RCVars["gSRTColor"] = mpResManager->getTexture("SRTReflection");

    mpGfxState->setFbo(mpRCFbo);
    mpRCShader->execute(pRenderContext, mpRCFbo);
}

void ReflectionCompositePass::renderGui(Gui::Window* pPassWindow)
{
    int dirty = 0;

    // Window is marked dirty if any of the configuration is changed.
    dirty |= (int)pPassWindow->checkbox("Skip RC computation", mSkipRC, false);

    // If any of our UI parameters changed, let the pipeline know we're doing something different next frame
    if (dirty) setRefreshFlag();
}

