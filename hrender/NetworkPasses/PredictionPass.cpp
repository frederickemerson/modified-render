#include "PredictionPass.h"
#include <cstdio>

namespace {
    // Where are our shaders located?
    const char* kPredictFragShader = "Samples\\hrender\\NetworkPasses\\Data\\NetworkPasses\\PredictionPass.ps.hlsl";
    const char* kOffsetBufferFragShader = "Samples\\hrender\\NetworkPasses\\Data\\NetworkPasses\\OffsetBuffer.ps.hlsl";
    const char* kCopyBufferFragShader = "Samples\\hrender\\NetworkPasses\\Data\\NetworkPasses\\CopyBuffer.ps.hlsl";

    // Size of camera data buffer
    const int camDataBufferSize = 50;
    // Minimum difference in number of frames before prediction occurs
    // i.e. Prediction does not occur when difference < threshhold
    //      Prediction occurs when difference >= threshhold
    const int predictionThreshhold = 0;

    // ============================= TEXTURES needed from ResourceManager ============================

    // Motion vectors resource location
    const std::string kMotionVec = "MotionVectors";
    // World position resource location
    const std::string kWorldPos = "WorldPosition";
    // Z-buffer resource
    const std::string kZBuffer = "Z-Buffer";
    // Original visibility buffer resource location
    const std::string kVisBufOrig = "VisibilityBitmap";
    // Offset visibility buffer resource location
    const std::string kVisBufOffset = "OffsetVisibilityBitmap";
    // Original AO buffer resource location
    const std::string kAOBufOrig = "AmbientOcclusion";
    // Offset AO buffer resource location
    const std::string kAOBufOffset = "OffsetAmbientOcclusion";
    // Original RCColor buffer resource location
    const std::string kRefBufOrig = "SRTReflection";
    // Offset RCColor buffer resource location
    const std::string kRefBufOffset = "OffsetSRTReflection";

    // ========================= SHADER VARIABLES FOR PredictionPass.ps.hlsl =========================

    // Name of shader variable for Z-buffer texture
    const std::string sVarZBufTex = "gZBuffer";
    // Name of shader variable for world position texture
    const std::string sVarWorldPosTex = "gWorldPos";

    // Name of shader variable for PredictionPass constant buffer
    const std::string sVarCBufferPrediction = "PredictionCb";

    // Name of shader variable for camera motion vector in world space
    const std::string sVarCamMotionVec = "gCamMotionVec";
    // Name of shader variable for camera's older view-projection matrix
    // (retrieved from the old camera data that was used to render the 
    // frame corresponding to the received visibility buffer)
    const std::string sVarOldVpMat = "gOldViewProjMat";
    // Name of shader variable for camera's near plane in camera space z-coordinates
    const std::string sVarCamNearZ = "gCamNearZ";
    // Name of shader variable for camera's far plane in camera space z-coordinates
    const std::string sVarCamFarZ = "gCamFarZ";
    // Name of shader variable for texture height in pixels
    const std::string sVarTexHeight = "gTexHeight";
    // Name of shader variable for texture width in pixels
    const std::string sVarTexWidth = "gTexWidth";


    // ========================== SHADER VARIABLES FOR OffsetBuffer.ps.hlsl ==========================

    // Name of shader variable for OffsetBuffer constant buffer
    const std::string sVarCBufferOffset = "OffsetCb";
    // Name of shader variable for texture height in pixels
    // const std::string sVarTexHeight = "gTexHeight";
    // Name of shader variable for texture width in pixels
    // const std::string sVarTexWidth = "gTexWidth";
    // Name of shader variable for offset factor
    const std::string sVarOffsetFactor = "gOffsetFactor";
    // Name of shader variable for original visibility buffer texture
    const std::string sVarVisBufOriginalTex = "gVisBufferOrig";
    // Name of shader variable for original visibility buffer texture
    const std::string sVarAOBufOriginalTex = "gAOBufferOrig";
    // Name of shader variable for original RayTracing Reflection buffer texture
    const std::string sVarRefBufOriginalTex = "gRefBufferOrig";

    // Name of shader variable for motion vectors texture from PredictionPass.ps.hlsl
    const std::string sVarMotionVecTex = "gMotionVectors";
    // Name of shader variable for mode of "filling-in" for unknown fragments
    // 0 - copy from old buffer
    // 1 - fill with shadow
    // 2 - fill with light
    const std::string sVarUnknownFrag = "gUnknownFragMode";

    // =========================== SHADER VARIABLES FOR CopyBuffer.ps.hlsl ===========================

    // Name of shader variable for original visibility buffer texture
    // const std::string sVarVisBufOriginalTex = "gVisBufferOrig";
}

PredictionPass::PredictionPass(int texWidth, int texHeight) :
    ::RenderPass("Prediction Pass", "Prediction Pass Options"),
    mTexWidth(texWidth), mTexHeight(texHeight), camDataBuffer(camDataBufferSize)
{}

bool PredictionPass::initialize(RenderContext* pRenderContext, ResourceManager::SharedPtr pResManager)
{
    // Stash a copy of our resource manager so we can get rendering resources
    mpResManager = pResManager;

    // Our GUI needs less space than other passes, so shrink the GUI window.
    setGuiSize(Falcor::int2(300, 70));

    mpPredictShader = FullscreenLaunch::create(kPredictFragShader);
    mpOffsetShader = FullscreenLaunch::create(kOffsetBufferFragShader);
    mpCopyShader = FullscreenLaunch::create(kCopyBufferFragShader);

    // Request a new texture for the motion vectors
    mMotionVecIndex = mpResManager->requestTextureResource(
        kMotionVec, Falcor::ResourceFormat::RGBA32Float, ResourceManager::kDefaultFlags, mTexWidth, mTexHeight);
    // Request for world positions
    mWorldPosIndex = mpResManager->requestTextureResource(
        kWorldPos, ResourceFormat::RGBA32Float, ResourceManager::kDefaultFlags, mTexWidth, mTexHeight);
    //// Request for original visibility buffer texture
    mVisBufOrigIndex = mpResManager->requestTextureResource(
        kVisBufOrig, ResourceFormat::R32Uint, ResourceManager::kDefaultFlags, mTexWidth, mTexHeight);
    // Request for offset visibility buffer texture
    mVisBufOffsetIndex = mpResManager->requestTextureResource(
        kVisBufOffset, ResourceFormat::R32Uint, ResourceManager::kDefaultFlags, mTexWidth, mTexHeight);
    // Request for original AO buffer texture
    mAOBufOrigIndex = mpResManager->requestTextureResource(
        kAOBufOrig, ResourceFormat::R32Uint, ResourceManager::kDefaultFlags, mTexWidth, mTexHeight);
    // Request for offset AO buffer texture
    mAOBufOffsetIndex = mpResManager->requestTextureResource(
        kAOBufOffset, ResourceFormat::R32Uint, ResourceManager::kDefaultFlags, mTexWidth, mTexHeight);
    // Request for original RayTracing Reflection buffer texture
    mRefBufOrigIndex = mpResManager->requestTextureResource(
        kRefBufOrig, ResourceFormat::R11G11B10Float, ResourceManager::kDefaultFlags, mTexWidth, mTexHeight);
    // Request for offset RayTracing Reflection buffer texture
    mRefBufOffsetIndex = mpResManager->requestTextureResource(
        kRefBufOffset, ResourceFormat::R11G11B10Float, ResourceManager::kDefaultFlags, mTexWidth, mTexHeight);

    // Request for the Z-Buffer
    mZBufferIndex = mpResManager->requestTextureResource(
        kZBuffer, Falcor::ResourceFormat::D24UnormS8, ResourceManager::kDepthBufferFlags, mTexWidth, mTexHeight);

    return true;
}

void PredictionPass::initScene(RenderContext* pRenderContext, Scene::SharedPtr pScene)
{
    // Stash a copy of the scene
    mpScene = pScene;
    if (!mpScene) return;
}

void PredictionPass::execute(Falcor::RenderContext* pRenderContext)
{
    // Get the CameraData for the current scene
    const Falcor::CameraData& currCamData = mpScene->getCamera()->getData();

    // Used for clearing output FBOs
    const Falcor::float4 clearColor(1, 0, 0, 1);
    // Create a framebuffer for rendering to, for the prediction step
    // (Creating once per frame is for simplicity, not performance)
    Falcor::Fbo::SharedPtr predictionFbo = mpResManager->
        createManagedFbo({ mMotionVecIndex }, mZBufferIndex);
    pRenderContext->clearFbo(predictionFbo.get(), clearColor, 1.0f, 0, FboAttachmentType::All);
    // Create a FBO for the step to offset the buffers
    Falcor::Fbo::SharedPtr offsetBufferFbo = mpResManager->
        createManagedFbo({ mVisBufOffsetIndex, mAOBufOffsetIndex, mRefBufOffsetIndex }, mZBufferIndex);
    //pRenderContext->clearFbo(offsetBufferFbo.get(), clearColor, 1.0f, 0, FboAttachmentType::All);
    // Failed to create valid FBOs? We're done.
    if (!(predictionFbo && offsetBufferFbo))
    {
        camDataBuffer.push_back(currCamData);
        return;
    }

    // Retrieve frames difference from NetworkManager
    int framesDiff = ResourceManager::mClientNetworkManager->numFramesBehind;

    mPercvDelay = framesDiff;
    char buffer[100];
    sprintf(buffer, "\nPerceived numFramesBehind is %d\n", framesDiff);
    OutputDebugStringA(buffer);

    int framesDifference = mUsePercvDelay ? mPercvDelay : mActualDelay;

    // Skip the pass if the difference in frames is greater
    // than the threshhold, or if it is larger than the
    // number of elements in the camera data circular buffer
    if (!(mUsePrediction && framesDifference >= predictionThreshhold &&
        framesDifference <= camDataBuffer.getNumberOfElements()))
    {
        camDataBuffer.push_back(currCamData);

        // Run a minimal shader to copy the received visibility buffer
        // to the one that will be used in the V-shading pass

        // Pass original visibility buffer to copy shader
        Falcor::GraphicsVars::SharedPtr copyShaderVars = mpCopyShader->getVars();
        copyShaderVars[sVarVisBufOriginalTex] = mpResManager->getTexture(mVisBufOrigIndex);
        copyShaderVars[sVarAOBufOriginalTex] = mpResManager->getTexture(mAOBufOrigIndex);
        copyShaderVars[sVarRefBufOriginalTex] = mpResManager->getTexture(mRefBufOrigIndex);

        // Using output FBO for OffsetBuffer, run CopyBuffer shader to copy over the visibility buffer
        mpCopyShader->execute(pRenderContext, offsetBufferFbo);
        return;
    }

    // Get CameraData that corresponds to the frame
    // of the incoming visibility buffer data scene
    // 
    // We use -framesDifference + 1, as the item in
    // the buffer at the 0th position will be 1
    // frame behind, the item at the -1st position
    // will be 2 frames behind, and so on
    const CameraData& prevCamData = framesDifference == 0
        ? currCamData
        : camDataBuffer.at(1 - framesDifference);

    // Store current camera data in the circular buffer
    camDataBuffer.push_back(currCamData);


    // We run the PredictionPass shader to get the motion vectors
    // Need to retrieve camera data of the old frame

    // Get old view-projection matrix of camera
    glm::float4x4 camOldViewProjMat = prevCamData.viewProjMat;
    // Get near and far plane of the current camera
    glm::float1 cameraNearZ = currCamData.nearZ;
    glm::float1 cameraFarZ = currCamData.farZ;

    // Pass camera data to shader, since FullscreenPass doesn't have scene information
    Falcor::GraphicsVars::SharedPtr predictShaderVars = mpPredictShader->getVars();
    predictShaderVars[sVarCBufferPrediction][sVarOldVpMat] = camOldViewProjMat;
    predictShaderVars[sVarCBufferPrediction][sVarCamNearZ] = cameraNearZ;
    predictShaderVars[sVarCBufferPrediction][sVarCamFarZ] = cameraFarZ;
    predictShaderVars[sVarCBufferPrediction][sVarTexHeight] = mpResManager->getHeight();
    predictShaderVars[sVarCBufferPrediction][sVarTexWidth] = mpResManager->getWidth();
    // Pass texture data to shader
    predictShaderVars[sVarWorldPosTex] = mpResManager->getTexture(mWorldPosIndex);

    // Using output FBO for PredictionPass, execute PredictionPass shader to get motion vectors
    mpPredictShader->execute(pRenderContext, predictionFbo);


    // Now we have the motion vectors in the correct texture
    // We run the OffsetBuffer shader to move the visibility buffer

    // Pass camera data to shader
    Falcor::GraphicsVars::SharedPtr offsetShaderVars = mpOffsetShader->getVars();
    offsetShaderVars[sVarCBufferOffset][sVarTexHeight] = mpResManager->getHeight();
    offsetShaderVars[sVarCBufferOffset][sVarTexWidth] = mpResManager->getWidth();
    offsetShaderVars[sVarCBufferOffset][sVarOffsetFactor] = mOffsetFactor;
    offsetShaderVars[sVarCBufferOffset][sVarUnknownFrag] = mUnknownFragmentsMode;
    offsetShaderVars[sVarVisBufOriginalTex] = mpResManager->getTexture(mVisBufOrigIndex);
    offsetShaderVars[sVarAOBufOriginalTex] = mpResManager->getTexture(mAOBufOrigIndex);
    offsetShaderVars[sVarRefBufOriginalTex] = mpResManager->getTexture(mRefBufOrigIndex);
    // Pass motion vectors texture to OffsetBuffer shader
    offsetShaderVars[sVarMotionVecTex] = mpResManager->getTexture(mMotionVecIndex);

    // Using output FBO for OffsetBuffer, execute OffsetBuffer shader and complete the full PredictionPass
    mpOffsetShader->execute(pRenderContext, offsetBufferFbo);
}

void PredictionPass::renderGui(Gui::Window* pPassWindow)
{
    // Set default window size
    if (mFirstGuiRender)
    {
        pPassWindow->windowSize(330, 120);
        mFirstGuiRender = false;
    }

    int dirty = 0;

    // Determine whether we are going prediction
    dirty |= (int)pPassWindow->checkbox(mUsePrediction ? "Applying prediction" : "No prediction", mUsePrediction);

    // Determine the extent of the offset
    // 0.0f is the minimum value of the offset and 2.0f is the maximum value
    dirty |= (int)pPassWindow->slider("Offset factor", mOffsetFactor, 0.0f, 0.1f);

    pPassWindow->text("Perceived number of frames behind: " + std::to_string(mPercvDelay));

    dirty |= (int)pPassWindow->checkbox(mUsePercvDelay ? "Using perceived delay" : "Not using perceived delay", mUsePercvDelay);
    if (!mUsePercvDelay)
    {
        dirty |= (int)pPassWindow->var("Delay to use for prediction", mActualDelay, 0, 1000, 0.01f);
    }

    pPassWindow->text("Unknown fragments mode");
    std::string unknownFragModeDisplay = "Fill with original buffer";
    if (mUnknownFragmentsMode == 1)
    {
        unknownFragModeDisplay = "Fill with shadow";
    }
    else if (mUnknownFragmentsMode == 2)
    {
        unknownFragModeDisplay = "Fill with illumination";
    }
    dirty |= (int)pPassWindow->var(unknownFragModeDisplay.c_str(), mUnknownFragmentsMode, 0, 2, 1);

    // If UI parameters change, let the pipeline know we're doing something different next frame
    if (dirty) setRefreshFlag();
}