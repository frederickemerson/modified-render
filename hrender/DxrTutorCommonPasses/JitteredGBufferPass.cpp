#include "JitteredGBufferPass.h"
#include <chrono>

namespace
{
    // Where is our environment map and scene located?
    const char *kEnvironmentMap = "MonValley_G_DirtRoad_3k.hdr";
    const char *kDefaultScene   = "pink_room\\pink_room.fscene";

    // For basic jittering, we don't need to change our rasterized g-buffer, just jitter the camera position
    const char *kGbufVertShader = "Samples\\hrender\\DxrTutorCommonPasses\\Data\\CommonPasses\\gBuffer.vs.hlsl";
    const char *kGbufFragShader = "Samples\\hrender\\DxrTutorCommonPasses\\Data\\CommonPasses\\gBuffer.ps.hlsl";
    const char *kClearToEnvMap  = "Samples\\hrender\\DxrTutorCommonPasses\\Data\\CommonPasses\\clearGBuffer.ps.hlsl";

    // If we want to jitter the camera to antialias using traditional a traditional 8x MSAA pattern, 
    //     use these positions (which are in the range [-8.0...8.0], so divide by 16 before use)
    const float kMSAA[8][2] = { {1,-3}, {-1,3}, {5,1}, {-3,-5}, {-5,5}, {-7,-1}, {3,7}, {7,-7} };
};

bool JitteredGBufferPass::initialize(RenderContext* pRenderContext, ResourceManager::SharedPtr pResManager)
{
    if (!pResManager) return false;

    // Stash a copy of our resource manager so we can get rendering resources
    mpResManager = pResManager;

    // We write to these output textures; tell our resource manager that we expect them to exist
    mpResManager->requestTextureResource("WorldPosition",       ResourceFormat::RGBA32Float);
    mpResManager->requestTextureResource("WorldNormal",         ResourceFormat::RGBA16Float);
    mpResManager->requestTextureResource("__TextureData",       ResourceFormat::RGBA32Float); // Stores 16 x uint8
    mpResManager->requestTextureResource("Z-Buffer",            ResourceFormat::D24UnormS8, ResourceManager::kDepthBufferFlags);

    // Set default environment map and scene
    mpResManager->updateEnvironmentMap(kEnvironmentMap);
    mpResManager->setDefaultSceneName(kDefaultScene);

    // Create our rasterization state and our raster shader wrapper
    mpGfxState = GraphicsState::create();

    // Create our wrapper for a scene-rasterization pass.
    mpRaster   = RasterLaunch::createFromFiles(kGbufVertShader, kGbufFragShader);

    // Create our wrapper for a full-screen raster pass to clear the g-buffer with the environment map
    mpClearGBuf = FullscreenLaunch::create(kClearToEnvMap);

    // Set up our random number generator by seeding it with the current time 
    auto currentTime    = std::chrono::high_resolution_clock::now();
    auto timeInMillisec = std::chrono::time_point_cast<std::chrono::milliseconds>(currentTime);
    mRng                = std::mt19937( uint32_t(timeInMillisec.time_since_epoch().count()) );
    
    return true;
}

void JitteredGBufferPass::initScene(RenderContext* pRenderContext, Scene::SharedPtr pScene)
{
    // Stash a copy of the scene.  Update our raster pass wrapper with the scene
    if (pScene) mpScene = pScene;
    if (mpRaster) mpRaster->setScene(mpScene);
}

void JitteredGBufferPass::renderGui(Gui::Window* pPassWindow)
{
    int dirty = 0;

    // Determine whether we're rendering the environment map
    dirty |= (int)pPassWindow->checkbox("Render environment map", mUseEnvMap);

    // Determine whether we're jittering at all
    dirty |= (int)pPassWindow->checkbox(mUseJitter ? "Using camera jitter" : "No camera jitter", mUseJitter);

    // Select what kind of jitter to use.  Right now, the choices are: 8x MSAA or completely random
    if (mUseJitter)
    {
        pPassWindow->text("     ");
        dirty |= (int)pPassWindow->checkbox(mUseRandom ? "Using randomized camera position" : "Using 8x MSAA pattern", mUseRandom, true);
    }

    // If UI parameters change, let the pipeline know we're doing something different next frame
    if (dirty) setRefreshFlag();
}

void JitteredGBufferPass::execute(RenderContext* pRenderContext)
{
    // Create a framebuffer for rendering.  (Creating once per frame is for simplicity, not performance).
    Fbo::SharedPtr outputFbo = mpResManager->createManagedFbo(
        { "WorldPosition", "WorldNormal", "__TextureData" }, 
        "Z-Buffer");                                                                                      

    // Failed to create a valid FBO?  We're done.
    if (!outputFbo) return;
    
    // Are we jittering?  If so, update our camera with the current jitter
    if (mUseJitter && mpScene && mpScene->getCamera())
    {
        // Increase our frame count
        mFrameCount++;

        // Determine our offset in the pixel in the range [-0.5...0.5]
        float xOff = mUseRandom ? mRngDist(mRng) - 0.5f : kMSAA[mFrameCount % 8][0]*0.0625f;
        float yOff = mUseRandom ? mRngDist(mRng) - 0.5f : kMSAA[mFrameCount % 8][1]*0.0625f;

        // Give our jitter to the scene camera
        mpScene->getCamera()->setJitter( xOff / float(outputFbo->getWidth()), yOff / float(outputFbo->getHeight()));
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
        // Pass our env. map to the shader (default one is filled with blue)
        clearGBufVars["gEnvMap"] = mpResManager->getTexture(ResourceManager::kEnvironmentMap);   
        // Pass camera data to shader, since FullscreenPass doesn't have scene information
        const CameraData& cameraData = mpScene->getCamera()->getData();
        clearGBufVars["CameraInfo"]["gCameraU"] = cameraData.cameraU;
        clearGBufVars["CameraInfo"]["gCameraV"] = cameraData.cameraV;
        clearGBufVars["CameraInfo"]["gCameraW"] = cameraData.cameraW;

        // Clear our framebuffer to the background environment map (and zeros elsewhere in the buffer)
        mpClearGBuf->execute(pRenderContext, outputFbo); 
    }
    
    // Execute our rasterization pass.  Note: Falcor will populate many built-in shader variables
    mpRaster->execute(pRenderContext, mpGfxState, outputFbo);
}
