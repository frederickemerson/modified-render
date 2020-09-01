#include "RasterLightingPass.h"

namespace
{
    // Lighting pass shader
    const char* kShaderFile = "Samples\\hrender\\RasterizedPasses\\Data\\rasterLighting.hlsl";
    // Shadow map pass shader
    const char* kShadowVertShader = "Samples\\hrender\\RasterizedPasses\\Data\\shadowmapPass.vs.hlsl";
    const char* kShadowFragShader = "Samples\\hrender\\RasterizedPasses\\Data\\shadowmapPass.ps.hlsl";

    // What are the entry points in that shader for various ray tracing shaders?
    // Note that this pass does not actually use ray tracing, we just use a raylaunch because of the convenient
    // access to full screen tex coords and the scene, which fullscreen launches do not have
    const char* kEntryPointRayGen = "RasterizationRayGen";

    const std::string kLightLocations = "lightLocations";

    // +x, -x, +y, -y, +z, -z vectors used to define the lookat matrices
    const float3 px = float3(1.0f, 0.0f, 0.0f);
    const float3 mx = float3(-1.0f, 0.0f, 0.0f);
    const float3 py = float3(0.0f, 1.0f, 0.0f);
    const float3 my = float3(0.0f, -1.0f, 0.0f);
    const float3 pz = float3(0.0f, 0.0f, 1.0f);
    const float3 mz = float3(0.0f, 0.0f, -1.0f);

    // Lookat/up vector directions for the right, left, top, bottom, back, front shadow map parts
    const float3 lookats[6] = { px, mx, py, my, pz, mz };
    // The up vectors are opposite of the typical cubemap convention, because falcor launch y-indices
    // are from top to bottom, which is reverse of typical y-indices
    const float3 ups[6] = { my, my, pz, mz, my, my };
}; 

bool RasterLightingPass::initialize(RenderContext* pRenderContext, ResourceManager::SharedPtr pResManager)
{
    // Stash a copy of our resource manager so we can get rendering resources
    mpResManager = pResManager;

    // Our GUI needs less space than other passes, so shrink the GUI window.
    setGuiSize(int2(300, 160));

    // Note that we some buffers from the G-buffer, plus the standard output buffer
    mpResManager->requestTextureResource("WorldPosition");
    mpResManager->requestTextureResource("WorldNormal");
    mpResManager->requestTextureResource("__TextureData");
    mOutputIndex = mpResManager->requestTextureResource(mOutputTexName);

    // Create our shadow map textures, which are arrays of 6 Texture2D each
    // This texture is used to store distance^2 of light to the object
    mpShadowMapTex = Texture::create2D(mShadowMapRes, mShadowMapRes, ResourceFormat::R32Float,
        6u, Texture::kMaxPossible, nullptr, ResourceManager::kDefaultFlags);
    // This texture is simply used for z-buffering
    mpShadowMapDepthTex = Texture::create2D(mShadowMapRes, mShadowMapRes, ResourceFormat::D24UnormS8,
        6u, Texture::kMaxPossible, nullptr, ResourceManager::kDepthBufferFlags );

    // Create our wrapper around a ray tracing pass. We don't actually do ray tracing, this is just
    // the most convenient way to perform a fullscreen pass while still having scene information.
    mpRays = RayLaunch::create(kShaderFile, kEntryPointRayGen);

    // Create our rasterization state and our raster shader wrapper for creating shadow maps
    mpGfxState = GraphicsState::create();
    mpShadowPass = RasterLaunch::createFromFiles(kShadowVertShader, kShadowFragShader);

    // Now that we've passed all our shaders in, compile and (if available) setup the scene
    if (mpScene) {
        mpRays->setScene(mpScene);
        mpRays->compileRayProgram();
        mpShadowPass->setScene(mpScene);
        // The size of the buffer includes the directional lights, and thus might be slightly oversized
        mLightPosBuffer = Buffer::createStructured(mpRays->getRayVars()[kLightLocations], (uint32_t)mpScene->getLightCount(),
            Resource::BindFlags::ShaderResource, Buffer::CpuAccess::None, nullptr, false);
    }

    // Create the shadow projection matrix, which will be updated whenever near/far are updated via the GUI
    mShadowProj = glm::perspective(glm::radians(90.0f), 1.0f, mShadowNear, mShadowFar);

    return true;
}

void RasterLightingPass::initScene(RenderContext* pRenderContext, Scene::SharedPtr pScene)
{
    // Stash a copy of the scene and pass it to our ray tracer (if initialized)
    mpScene = pScene;
    if (!mpScene) return;
    if (mpRays) {
        mpRays->setScene(mpScene);
        mpRays->compileRayProgram();
        mpShadowPass->setScene(mpScene);

        // The size of the buffer includes the directional lights, and thus might be slightly oversized
        mLightPosBuffer = Buffer::createStructured(mpRays->getRayVars()[kLightLocations], (uint32_t)mpScene->getLightCount(), Resource::BindFlags::ShaderResource, Buffer::CpuAccess::None, nullptr, false);
    }
}

void RasterLightingPass::execute(RenderContext* pRenderContext)
{
    // Get the output buffer we're writing into
    Texture::SharedPtr pDstTex = mpResManager->getClearedTexture(mOutputIndex, float4(0.0f));

    // Do we have all the resources we need to render?  If not, return
    if (!pDstTex || !mpRays || !mpRays->readyToRender()) return;

    // Set our lighting pass shader variables 
    auto rayVars = mpRays->getRayVars();
    rayVars["gPos"]       = mpResManager->getTexture("WorldPosition");
    rayVars["gNorm"]      = mpResManager->getTexture("WorldNormal");
    rayVars["gTexData"]   = mpResManager->getTexture("__TextureData");
    rayVars["gShadowMap"] = mpShadowMapTex;
    rayVars["gOutput"]    = pDstTex;
    rayVars["RasterBuf"]["gShadowMapRes"] = mShadowMapRes;
    rayVars["RasterBuf"]["gShadowFar2"]   = mShadowFar * mShadowFar;
    rayVars["RasterBuf"]["gShadowOffset"] = mShadowOffset;
    rayVars["RasterBuf"]["gShowLights"]   = mShowLights;

    // If mShowLights is true, we will draw markers on the screen where the lights are.
    // Compute and send the light locations to the shader.
    if (mShowLights) drawDebugLights();

    // We need to perform the lighting pass for the scene once per light.
    // Alternative is to send all shadow maps to the GPU, which means creating multiple shadow
    // textures. Trade space for time.
    // TODO: Change implementation to that.
    uint32_t lightCount = mpScene->getLightCount();
    for (uint32_t lightIdx = 0; lightIdx < lightCount; lightIdx++)
    {
        Light::SharedPtr currLight = mpScene->getLight(lightIdx);

        // Only use shadow mapping for the point lights for now
        bool usingShadowMapping = currLight->getType() == LightType::Point;
        // Populate the shadow map for this light by rendering the scene from 6 views 
        if (usingShadowMapping) fillShadowMap(pRenderContext, currLight);

        // Let the shader know which light it is computing lighting for, and whether shadow
        // mapping is being used.
        rayVars["RasterBuf"]["gCurrLightIdx"] = lightIdx;
        rayVars["RasterBuf"]["gUsingShadowMapping"] = usingShadowMapping;

        // Perform lighting pass for this light
        mpRays->execute(pRenderContext, uint2(pDstTex->getWidth(), pDstTex->getHeight()));
    }
}

void RasterLightingPass::fillShadowMap(Falcor::RenderContext* pRenderContext, Light::SharedPtr currLight)
{
    // Get the parameter block to send variables to the shader
    auto shadowVars = mpShadowPass->getVars();

    // Clear the "depth" buffer to 1.0f, and actual depth buffer
    gpFramework->getRenderContext()->clearRtv(mpShadowMapTex->getRTV().get(), uint4(1));
    gpFramework->getRenderContext()->clearDsv(mpShadowMapDepthTex->getDSV().get(), 1.0f, 0);

    // Pixel shader variable that doesn't change between passes
    shadowVars["ShadowPsVars"]["gShadowFar2"] = mShadowFar * mShadowFar;

    // Render the scene for each of the 6 faces of the shadow map.
    // TODO: Change this to be done in a geometry shader in future to minimize CPU->GPU calls
    for (int face = 0; face < 6; face++)
    {
        float3 currLightPos = currLight->getData().posW;

        // Create the view projection matrix from the perspective of the light for this face of the cubemap
        glm::mat4 shadowViewProj = mShadowProj * glm::lookAt(currLightPos, currLightPos + lookats[face], ups[face]);

        // Vertex shader variables
        shadowVars["ShadowMapBuf"]["gLightViewProj"] = shadowViewProj;

        // Pixel shader variables
        shadowVars["ShadowPsVars"]["gLightPosW"] = currLightPos;

        // Create an FBO using the specific perspective being rendered
        auto pTargetFbo = Fbo::create();
        pTargetFbo->attachColorTarget(mpShadowMapTex, 0, 0u, face);
        pTargetFbo->attachDepthStencilTarget(mpShadowMapDepthTex, 0u, face);

        // Render the scene from this perspective
        mpShadowPass->execute(pRenderContext, mpGfxState, pTargetFbo);
    }
}

void RasterLightingPass::drawDebugLights()
{
    auto rayVars = mpRays->getRayVars();

    // TODO: This is the buffer to be sent to GPU. Does not really need to be computed per frame,
    // we should use a dirty bit to do that.
    std::vector<float3> lightPositions;
    // The number of point lights to be sent to the shader, so that the correct number of lights are rendered.
    // Cannot be precomputed, since we do CPU clipping of the lights.
    uint32_t numPointLights = 0;

    const glm::mat4& camViewProj = mpScene->getCamera()->getViewProjMatrix();
    uint32_t lightCount = mpScene->getLightCount();
    for (uint32_t i = 0; i < lightCount; i++)
    {
        Light::SharedPtr currLight = mpScene->getLight(i);

        // We only render point lights
        if (currLight->getType() != LightType::Point) continue;

        // Get the light's world position
        float3 posw = currLight->getData().posW;
        // We transpose because we need to pre-multiply the column-major matrix. 
        float4 posh = float4(posw, 1.0f) * glm::transpose(camViewProj);
        // Perform perspective division (divide x, y, z by w, where w increases the further the object is)
        // This gives us clip coordinates ([-1.0, 1.0] NDC)
        float3 posh_ndc = posh / posh[3];

        // Clip the light if it's out of view
        if (fabs(posh_ndc.x) >= 1 || fabs(posh_ndc.y) >= 1 || posh_ndc.z >= 1) continue;

        // Convert from NDC to window coordinates [0, 1.0]. We flip the y-axis because the y-indices in Falcor are flipped.
        // To render the light positions in the shader, we need to put this into viewport space by multiplying
        // the dimensions of the viewport.
        float3 pos_window = float3(posh_ndc[0], posh_ndc[1] * (-1.f), posh_ndc[2]) / 2.f + .5f;

        // Store this in the vector that will be sent to the GPU
        lightPositions.push_back(pos_window);
        numPointLights++;
    }

    // Send light positions to the shader if there is at least one
    if (numPointLights > 0)
    {
        mLightPosBuffer->setBlob(&lightPositions[0], 0, numPointLights * sizeof(float3));
        rayVars->setBuffer(kLightLocations, mLightPosBuffer);
    }

    // Send other variables to shader
    rayVars["RasterBuf"]["gNumPointLights"] = numPointLights;
    rayVars["RasterBuf"]["gLightRadius"] = mLightRadius;
}

void RasterLightingPass::renderGui(Gui::Window* pPassWindow)
{
    int dirty = 0;

    dirty |= (int)pPassWindow->checkbox("Debug Light Positions", mShowLights, false);
    dirty |= (int)pPassWindow->var("Light Radius", mLightRadius, 0.01f, FLT_MAX, 0.1f, false);
    dirty |= (int)pPassWindow->var("ShadowMap Offset", mShadowOffset, 0.000001f, FLT_MAX, 0.000005f, false);

    int shadowChanged = (int)pPassWindow->var("ShadowMap Near", mShadowNear, 0.01f, FLT_MAX, 0.1f, false);
    shadowChanged |= (int)pPassWindow->var("ShadowMap Far", mShadowFar, 0.01f, FLT_MAX, 0.1f, false);
    // If near/far values are updated, recreate the shadow projection matrix
    if (shadowChanged)
    {
        mShadowProj = glm::perspective(glm::radians(90.0f), 1.0f, mShadowNear, mShadowFar);
        dirty |= 1;
    }

    // If any of our UI parameters changed, let the pipeline know we're doing something different next frame
    if (dirty) setRefreshFlag();
}
