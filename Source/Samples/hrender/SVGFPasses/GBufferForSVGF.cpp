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

#include "GBufferForSVGF.h"

namespace
{
    // Where is our environment map and scene located?
    const char *kEnvironmentMap = "MonValley_G_DirtRoad_3k.hdr";
    const char *kDefaultScene   = "pink_room\\pink_room.fscene";

    // Where are our shaders located?
    const char *kGbufVertShader = "Samples\\hrender\\SVGFPasses\\Data\\SVGFPasses\\gBufferSVGF.vs.hlsl";
    const char *kGbufFragShader = "Samples\\hrender\\SVGFPasses\\Data\\SVGFPasses\\gBufferSVGF.ps.hlsl";
    const char *kClearToEnvMap  = "Samples\\hrender\\SVGFPasses\\Data\\SVGFPasses\\clearGBuffer.ps.hlsl";

    // If we want to jitter the camera to antialias using traditional a traditional 8x MSAA pattern, 
    //     use these positions (which are in the range [-8.0...8.0], so divide by 16 before use)
    const float kMSAA[8][2] = { {1,-3}, {-1,3}, {5,1}, {-3,-5}, {-5,5}, {-7,-1}, {3,7}, {7,-7} };
};

bool GBufferForSVGF::initialize(RenderContext* pRenderContext, ResourceManager::SharedPtr pResManager)
{
    if (!pResManager) return false;

    // Stash a copy of our resource manager so we can get rendering resources
    mpResManager = pResManager;

    // We write these texture; tell our resource manager that we expect these channels to exist
    mpResManager->requestTextureResource("WorldPosition",           ResourceFormat::RGBA32Float);
    mpResManager->requestTextureResource("WorldNormal",             ResourceFormat::RGBA16Float);
    mpResManager->requestTextureResource("__TextureData",           ResourceFormat::RGBA32Float);
    mpResManager->requestTextureResource("SVGF_LinearZ",            ResourceFormat::RGBA32Float);
    mpResManager->requestTextureResource("SVGF_MotionVecs",         ResourceFormat::RGBA16Float);
    mpResManager->requestTextureResource("SVGF_CompactNormDepth",   ResourceFormat::RGBA32Float);
    mpResManager->requestTextureResource("Z-Buffer",                ResourceFormat::D24UnormS8, ResourceManager::kDepthBufferFlags);

    // Set default environment map and scene
    mpResManager->updateEnvironmentMap(kEnvironmentMap);
    mpResManager->setDefaultSceneName(kDefaultScene);

    // If the user loads an environment map, grab it here (to display in g-buffer)
    mpResManager->requestTextureResource(ResourceManager::kEnvironmentMap);

    // Since we're rasterizing, we need to define our raster pipeline state (though we use the defaults)
    mpGfxState = GraphicsState::create();

    // Create our wrapper for a scene-rasterization pass.
    mpRaster = RasterLaunch::createFromFiles(kGbufVertShader, kGbufFragShader);

    // Create our wrapper for a full-screen raster pass to clear the g-buffer with the environment map
    mpClearGBuf = FullscreenLaunch::create(kClearToEnvMap);

    // Set up our random number generator by seeding it with the current time 
    auto currentTime = std::chrono::high_resolution_clock::now();
    auto timeInMillisec = std::chrono::time_point_cast<std::chrono::milliseconds>(currentTime);
    mRng = std::mt19937(uint32_t(timeInMillisec.time_since_epoch().count()));

    return true;
}

void GBufferForSVGF::initScene(RenderContext* pRenderContext, Scene::SharedPtr pScene)
{
    // Stash a copy of the scene
    if (pScene) 
        mpScene = pScene;

    // Update our raster pass wrapper with this scene
    if (mpRaster) 
        mpRaster->setScene(mpScene);
}

void GBufferForSVGF::renderGui(Gui::Window* pPassWindow)
{
    int dirty = 0;

    // Determine whether we're rendering the environment map
    dirty |= (int)pPassWindow->checkbox("Render environment map", mUseEnvMap);

    // Determine whether we're jittering at all
    dirty |= (int)pPassWindow->checkbox("Enable camera jitter", mUseJitter);

    // Select what kind of jitter to use.  Right now, the choices are: 8x MSAA or completely random
    if (mUseJitter)
    {
        pPassWindow->text("     ");
        dirty |= (int)pPassWindow->checkbox(mUseRandom ? "Using randomized camera position" : "Using 8x MSAA pattern", mUseRandom, true);
    }

    // If UI parameters change, let the pipeline know we're doing something different next frame
    if (dirty) setRefreshFlag();
}

void GBufferForSVGF::execute(RenderContext* pRenderContext)
{
    // Create a framebuffer for rendering.  (Creating once per frame is for simplicity, not performance).
    Fbo::SharedPtr outputFbo = mpResManager->createManagedFbo(
        { "WorldPosition", "WorldNormal", "__TextureData", "SVGF_LinearZ", "SVGF_MotionVecs", "SVGF_CompactNormDepth" }, 
        "Z-Buffer" );                                                                                     

    // Failed to create a valid FBO?  We're done.
    if (!outputFbo) return;

    // Are we jittering?  If so, update our camera with the current jitter
    if (mUseJitter && mpScene && mpScene->getCamera())
    {
        // Increase our frame count
        mFrameCount++;

        // Determine our offset in the pixel in the range [-0.5...0.5]
        float xOff = mUseRandom ? mRngDist(mRng) - 0.5f : kMSAA[mFrameCount % 8][0] * 0.0625f;
        float yOff = mUseRandom ? mRngDist(mRng) - 0.5f : kMSAA[mFrameCount % 8][1] * 0.0625f;

        // Give our jitter to the scene camera
        mpScene->getCamera()->setJitter(xOff / float(outputFbo->getWidth()), yOff / float(outputFbo->getHeight()));
    }

    // If we're not using environment map, we clear the FBO regularly
    if (!mUseEnvMap)
    {
        // Clear our g-buffer.  All color buffers to (0,0,0,0), depth to 1, stencil to 0
        pRenderContext->clearFbo(outputFbo.get(), float4(0, 0, 0, 0), 1.0f, 0);

        // Separately clear our diffuse color buffer to the background color, rather than black
        pRenderContext->clearUAV(outputFbo->getColorTexture(2)->getUAV().get(), float4(mBgColor, 1.0f));
    }
    // We are using the environment map, which we fill with a fullscreen pass
    else
    {
        // Clear our g-buffer's depth buffer (clear depth to 1, stencil to 0)
        pRenderContext->clearDsv(outputFbo->getDepthStencilView().get(), 1.0f, 0);

        auto clearGBufVars = mpClearGBuf->getVars();
        // Clear our framebuffer to the background environment map (and zeros elsewhere in the buffer)
        clearGBufVars["gEnvMap"] = mpResManager->getTexture(ResourceManager::kEnvironmentMap);   // Get our env. map (default one is filled with blue)
        // Pass camera data to shader, since FullscreenPass doesn't have scene information
        const CameraData& cameraData = mpScene->getCamera()->getData();
        clearGBufVars["CameraInfo"]["gCameraU"] = cameraData.cameraU;
        clearGBufVars["CameraInfo"]["gCameraV"] = cameraData.cameraV;
        clearGBufVars["CameraInfo"]["gCameraW"] = cameraData.cameraW;

        // Clear our framebuffer to the background environment map (and zeros elsewhere in the buffer)
        mpClearGBuf->execute(pRenderContext, outputFbo);
    }

    // Pass down our output size to the G-buffer shader
    auto shaderVars = mpRaster->getVars();
    float2 fboSize = float2(outputFbo->getWidth(), outputFbo->getHeight());
    shaderVars["GBufCB"]["gBufSize"] = float4(fboSize.x, fboSize.y, 1.0f / fboSize.x, 1.0f / fboSize.y);

    // Execute our rasterization pass.  Note: Falcor will populate many built-in shader variables
    mpRaster->execute(pRenderContext, mpGfxState, outputFbo);
}
