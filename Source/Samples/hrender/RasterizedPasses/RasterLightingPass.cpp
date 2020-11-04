#include "RasterLightingPass.h"

namespace
{
    // Lighting pass shader
    const char* kShaderFile = "Samples\\hrender\\RasterizedPasses\\Data\\rasterLighting.hlsl";
    // Shadow map pass shader
    const char* kShadowVertShader = "Samples\\hrender\\RasterizedPasses\\Data\\shadowmapPass.vs.hlsl";
    const char* kShadowFragShader = "Samples\\hrender\\RasterizedPasses\\Data\\shadowmapPass.ps.hlsl";
    const char* kShadowCubeFragShader = "Samples\\hrender\\RasterizedPasses\\Data\\shadowmapCubePass.ps.hlsl";

    // What are the entry points in that shader for various ray tracing shaders?
    // Note that this pass does not actually use ray tracing, we just use a raylaunch because of the convenient
    // access to full screen tex coords and the scene, which fullscreen launches do not have
    const char* kEntryPointRayGen = "RasterizationRayGen";

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

    setGuiSize(int2(300, 250));

    // Note that we some buffers from the G-buffer, plus the standard output buffer
    mpResManager->requestTextureResource("WorldPosition");
    mpResManager->requestTextureResource("WorldNormal");
    mpResManager->requestTextureResource("__TextureData");
    mOutputIndex = mpResManager->requestTextureResource(mOutputTexName);

    // Create our wrapper around a ray tracing pass. We don't actually do ray tracing, this is just
    // the most convenient way to perform a fullscreen pass while still having scene information.
    mpRays = RayLaunch::create(kShaderFile, kEntryPointRayGen);

    // Create our rasterization state and our raster shader wrapper for creating shadow maps
    mpCubeGfxState = GraphicsState::create();
    mpDirGfxState = GraphicsState::create();
    updateCullMode(); // Enable front face culling by default

    // For the directional map, uncomment this to disable culling (for indoor roof type things)
    //{
    //    RasterizerState::Desc rsDesc;
    //    rsDesc.setCullMode(RasterizerState::CullMode::None);
    //    mpDirGfxState->setRasterizerState(RasterizerState::create(rsDesc));
    //}

    mpShadowPass = RasterLaunch::createFromFiles(kShadowVertShader, kShadowFragShader);
    mpShadowCubePass = RasterLaunch::createFromFiles(kShadowVertShader, kShadowCubeFragShader);

    // Now that we've passed all our shaders in, compile and (if available) setup the scene
    if (mpScene) initScene(pRenderContext, mpScene);

    // Create the shadow projection matrix, which will be updated whenever near/far are updated via the GUI
    mCubeShadowProj = glm::perspective(glm::radians(90.0f), 1.0f, mCubeShadowNear, mCubeShadowFar);

    return true;
}

void RasterLightingPass::initScene(RenderContext* pRenderContext, Scene::SharedPtr pScene)
{
    // Stash a copy of the scene and pass it to our ray tracer (if initialized)
    mpScene = pScene;
    if (!mpScene) return;
    if (mpRays)
    {
        mpRays->setScene(mpScene);
        mpRays->compileRayProgram();
        mpShadowPass->setScene(mpScene);
        mpShadowCubePass->setScene(mpScene);

        // Count the number of point/directional lights in the scene to create shadow cubemaps/directional shadow maps
        mNumPointLights = 0; mNumDirLights = 0;
        for (uint i = 0; i < mpScene->getLightCount(); i++)
        {
            Light::SharedPtr currLight = mpScene->getLight(i);
            mNumPointLights += currLight->getType() == LightType::Point ? 1 : 0;
            // We actually will only render light from one directional light - scenes shouldn't have more than one.
            mNumDirLights += currLight->getType() == LightType::Directional ? 1 : 0;
        }

        if (mNumPointLights > 0)
        {
            // Create the array of point light positions for drawing of debug lights
            mLightPosBuffer = Buffer::createStructured(mpRays->getRayVars()["gLightLocations"],
                mNumPointLights, Resource::BindFlags::ShaderResource, Buffer::CpuAccess::None, nullptr, false);

            // Create our shadow map textures for each point light, which are arrays of Texture2D of 6*resolution each
            // This texture is used to store distance^2 of light to the object
            mpCubeShadowMapTex = Texture::create2D(mCubeShadowMapRes, mCubeShadowMapRes * 6, ResourceFormat::R32Float,
                mNumPointLights, Texture::kMaxPossible, nullptr, ResourceManager::kDefaultFlags);
            // This texture is simply used for z-buffering
            mpCubeShadowMapZTex = Texture::create2D(mCubeShadowMapRes, mCubeShadowMapRes * 6, ResourceFormat::D24UnormS8,
                mNumPointLights, Texture::kMaxPossible, nullptr, ResourceManager::kDepthBufferFlags);
        }
        if (mNumDirLights > 0)
        {
            // Create our shadow map textures for the directional light, which are arrays of NUM_CASCADES Texture2D
            // We assume there is only one directional light in a scene.
            // This texture is used to store distance^2 of light to the object
            mpDirShadowMapTex = Texture::create2D(mDirShadowMapRes, mDirShadowMapRes, ResourceFormat::R32Float,
                NUM_CASCADES, Texture::kMaxPossible, nullptr, ResourceManager::kDefaultFlags);
            // This texture is simply used for z-buffering
            mpDirShadowMapZTex = Texture::create2D(mDirShadowMapRes, mDirShadowMapRes, ResourceFormat::D24UnormS8,
                NUM_CASCADES, Texture::kMaxPossible, nullptr, ResourceManager::kDepthBufferFlags);
            mDirShadowOriginBuffer = Buffer::createStructured(mpRays->getRayVars()["gDirOrigins"],
                NUM_CASCADES, Resource::BindFlags::ShaderResource, Buffer::CpuAccess::None, nullptr, false);
            mDirShadowRadiusBuffer = Buffer::createStructured(mpRays->getRayVars()["gDirRadii"],
                NUM_CASCADES, Resource::BindFlags::ShaderResource, Buffer::CpuAccess::None, nullptr, false);
            mDirShadowViewProjBuffer = Buffer::createStructured(mpRays->getRayVars()["gDirViewProj"],
                NUM_CASCADES, Resource::BindFlags::ShaderResource, Buffer::CpuAccess::None, nullptr, false);
        }
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
    rayVars["gPos"]           = mpResManager->getTexture("WorldPosition");
    rayVars["gNorm"]          = mpResManager->getTexture("WorldNormal");
    rayVars["gTexData"]       = mpResManager->getTexture("__TextureData");
    rayVars["gDirShadowMap"]  = mpDirShadowMapTex;
    rayVars["gCubeShadowMap"] = mpCubeShadowMapTex;
    rayVars["gOutput"]        = pDstTex;
    rayVars["RasterBuf"]["gUsingShadowMapping"] = mUsingShadowMapping;
    rayVars["RasterBuf"]["gDirShadowMapRes"]    = mDirShadowMapRes;
    rayVars["RasterBuf"]["gDirAmbient"]         = mDirAmbient;
    rayVars["RasterBuf"]["gDirShadowBias"]      = mDirShadowBias;
    rayVars["RasterBuf"]["gCubeShadowMapRes"]   = mCubeShadowMapRes;
    rayVars["RasterBuf"]["gCubeShadowFar2"]     = mCubeShadowFar * mCubeShadowFar;
    rayVars["RasterBuf"]["gCubeShadowBias"]     = mCubeShadowBias;
    rayVars["RasterBuf"]["gNumCascades"]        = NUM_CASCADES;
    // Debug light/cascade drawing
    rayVars["RasterBuf"]["gShowLights"]         = mShowLights;
    rayVars["RasterBuf"]["gShowCascades"]         = mShowCascades;
    // PCF options
    rayVars["RasterBuf"]["gUsingPCF"]           = mUsingPCF;
    rayVars["RasterBuf"]["gCubePCFWidth"]       = mCubePCFWidth;
    rayVars["RasterBuf"]["gDirPCFWidth"]        = mDirPCFWidth;

    // If mShowLights is true, we will draw markers on the screen where the lights are.
    // Compute and send the light locations to the shader.
    if (mShowLights) drawDebugLights();

    if (is_set(mpScene->getUpdates(), Scene::UpdateFlags::All))
    {
        // Clear the shadow cube "depth" buffers to 1.0f, and actual depth buffers
        if (mNumPointLights > 0)
        {
            gpFramework->getRenderContext()->clearRtv(mpCubeShadowMapTex->getRTV().get(), uint4(1));
            gpFramework->getRenderContext()->clearDsv(mpCubeShadowMapZTex->getDSV().get(), 1.0f, 0);
        }
        // Clear the directional shadow "depth" buffers to 1.0f, and actual depth buffers
        if (mNumDirLights > 0)
        {
            gpFramework->getRenderContext()->clearRtv(mpDirShadowMapTex->getRTV().get(), uint4(1));
            gpFramework->getRenderContext()->clearDsv(mpDirShadowMapZTex->getDSV().get(), 1.0f, 0);
        }

        // We need to perform the shadowmap filling pass for the scene once per light.
        uint32_t lightCount = mpScene->getLightCount();
        uint32_t pointLightIdx = 0, dirLightIdx = 0; // dirLightIdx is not yet being used
        for (uint32_t lightIdx = 0; lightIdx < lightCount; lightIdx++)
        {
            Light::SharedPtr currLight = mpScene->getLight(lightIdx);

            if (mUsingShadowMapping && currLight->getType() == LightType::Point)
            {
                // Populate the shadow map for this light by rendering the scene from 6 views,
                // but only if the light uses shadow mapping, and we enabled shadow mapping.
                fillShadowCubeMap(pRenderContext, currLight, pointLightIdx++);
            }
            if (mUsingShadowMapping && currLight->getType() == LightType::Directional)
            {
                fillShadowDirMap(pRenderContext, currLight);
            }
        }
    }

    // Perform lighting pass for this light
    mpRays->execute(pRenderContext, uint2(pDstTex->getWidth(), pDstTex->getHeight()));
}

void RasterLightingPass::fillShadowCubeMap(Falcor::RenderContext* pRenderContext, Light::SharedPtr currLight, uint pointLightIdx)
{
    // Get the parameter block to send variables to the shader
    auto shadowVars = mpShadowCubePass->getVars();

    // Pixel shader variable that doesn't change between passes
    shadowVars["ShadowPsVars"]["gShadowFar2"] = mCubeShadowFar * mCubeShadowFar;
    shadowVars["ShadowPsVars"]["gCubeShadowMapRes"] = mCubeShadowMapRes;

    // Render the scene for each of the 6 faces of the shadow map.
    // TODO: Change this to be done in a geometry shader in future to minimize CPU->GPU calls, and
    //       do some profiling to see if that improves performance (not guaranteed)
    for (int face = 0; face < 6; face++)
    {
        float3 currLightPos = currLight->getData().posW;

        // Create the view projection matrix from the perspective of the light for this face of the cubemap
        glm::mat4 shadowViewProj = mCubeShadowProj * glm::lookAt(currLightPos, currLightPos + lookats[face], ups[face]);

        // Vertex shader variables
        shadowVars["ShadowMapBuf"]["gLightViewProj"] = shadowViewProj;
        // Pixel shader variables
        shadowVars["ShadowPsVars"]["gLightPosW"] = currLightPos;
        shadowVars["ShadowPsVars"]["gFace"] = face;

        // Create an FBO using the specific perspective being rendered
        auto pTargetFbo = Fbo::create();
        pTargetFbo->attachColorTarget(mpCubeShadowMapTex, 0, 0u, pointLightIdx);
        pTargetFbo->attachDepthStencilTarget(mpCubeShadowMapZTex, 0u, pointLightIdx);

        GraphicsState::Viewport currViewport(0.0f, (float)face * mCubeShadowMapRes,
                                             (float)mCubeShadowMapRes, (float)mCubeShadowMapRes, 0.0f, 1.0f);
        mpCubeGfxState->setViewport(0, currViewport);
        auto x = mpCubeGfxState->getViewports();
        // Render the scene from this perspective
        mpShadowCubePass->execute(pRenderContext, mpCubeGfxState, pTargetFbo, false);
    }
}

void RasterLightingPass::fillShadowDirMap(Falcor::RenderContext* pRenderContext, Light::SharedPtr currLight)
{
    // Get the parameter block to send variables to the shader
    auto shadowVars = mpShadowPass->getVars();

    float3 lightDir = ((DirectionalLight*)currLight.get())->getWorldDirection();

    // Store the maximum frustum radius
    float maxRadius = 0.0f;
    // Compute origins and radii of all the cascades
    for (uint i = 0; i < NUM_CASCADES; i++)
    {
        // Get world space bounding coordinates of the camera's view
        camClipSpaceToWorldSpace(mpScene->getCamera().get(), mDirShadowOrigin[i], mDirShadowRadius[i], i);
        maxRadius = std::max(maxRadius, mDirShadowRadius[i]);
    }

    // Compute view-projection matrices and render the shadow maps for each cascade
    for (uint i = 0; i < NUM_CASCADES; i++)
    {
        // Create directional shadow map
        glm::mat4 view = glm::lookAt(mDirShadowOrigin[i], mDirShadowOrigin[i] + lightDir, float3(0.f, 1.f, 0.f));
        glm::mat4 proj = glm::ortho(-mDirShadowRadius[i], mDirShadowRadius[i],
                                    -mDirShadowRadius[i], mDirShadowRadius[i],
                                    -maxRadius, maxRadius);
        mDirShadowViewProj[i] = proj * view;
        // Render the scene with the computed matrix
        // Vertex shader variables
        shadowVars["ShadowMapBuf"]["gLightViewProj"] = mDirShadowViewProj[i];

        // Create an FBO using the specific perspective being rendered
        auto pTargetFbo = Fbo::create();
        pTargetFbo->attachColorTarget(mpDirShadowMapTex, 0, 0u, i);
        pTargetFbo->attachDepthStencilTarget(mpDirShadowMapZTex, 0u, i);

        // Render the scene from this perspective
        mpShadowPass->execute(pRenderContext, mpDirGfxState, pTargetFbo);
    }

    // Lighting pass variables that were computed for the shadowmap's cascades
    auto shadingVars = mpRays->getRayVars();
    // The computed shadow view-projection matrices for each cascade
    mDirShadowViewProjBuffer->setBlob(mDirShadowViewProj, 0, NUM_CASCADES * sizeof(glm::mat4));
    shadingVars->setBuffer("gDirViewProj", mDirShadowViewProjBuffer);
    // The bounding radius of each cascade
    mDirShadowRadiusBuffer->setBlob(mDirShadowRadius, 0, NUM_CASCADES * sizeof(float));
    shadingVars->setBuffer("gDirRadii", mDirShadowRadiusBuffer);
    // The origin of each cascade
    mDirShadowOriginBuffer->setBlob(mDirShadowOrigin, 0, NUM_CASCADES * sizeof(float3));
    shadingVars->setBuffer("gDirOrigins", mDirShadowOriginBuffer);
}

void RasterLightingPass::drawDebugLights()
{
    auto rayVars = mpRays->getRayVars();

    if (is_set(mpScene->getUpdates(), Scene::UpdateFlags::All))
    {
        std::vector<float3> lightPositions;
        // The number of point lights to be sent to the shader, so that the correct number of lights are rendered.
        // Cannot be precomputed, since we do CPU clipping of the lights.
        mNumPointLightsVisible = 0;

        const glm::mat4& camViewProj = mpScene->getCamera()->getViewProjMatrix();
        uint32_t lightCount = mpScene->getLightCount();
        for (uint32_t i = 0; i < lightCount; i++)
        {
            Light::SharedPtr currLight = mpScene->getLight(i);

            // We only render point lights
            if (currLight->getType() != LightType::Point) continue;

            // Get the light's world position
            float3 posw = currLight->getData().posW;
            float4 posh = camViewProj * float4(posw, 1.0f);
            // Perform perspective division (divide x, y, z by w, where w increases the further the object is)
            // This gives us clip coordinates ([-1.0, 1.0] NDC)
            float3 posh_ndc = posh / posh[3];

            // Clip the light if it's out of view
            if (fabs(posh_ndc.x) >= 1 || fabs(posh_ndc.y) >= 1 || posh_ndc.z >= 1) continue;

            // Convert from NDC to window coordinates [0, 1.0]. We flip the y-axis because the y-indices in Falcor are flipped.
            // To render the light positions in the shader, we need to put this into viewport space by multiplying
            // the dimensions of the viewport.
            float3 pos_window = float3(posh_ndc[0], posh_ndc[1] * (-1.f), posh_ndc[2]) * 0.5f + .5f;

            // Store this in the vector that will be sent to the GPU
            lightPositions.push_back(pos_window);
            mNumPointLightsVisible++;
        }

        // Update the buffer if we have visible lights
        if (mNumPointLightsVisible > 0)
        {
            mLightPosBuffer->setBlob(&lightPositions[0], 0, mNumPointLightsVisible * sizeof(float3));
        }
    }

    // Send other variables to shader
    // Only send light positions to the shader if there is at least one
    if (mNumPointLightsVisible > 0)
    {
        rayVars->setBuffer("gLightLocations", mLightPosBuffer);
    }
    rayVars["RasterBuf"]["gNumPointLights"] = mNumPointLightsVisible;
    rayVars["RasterBuf"]["gLightRadius"] = mLightRadius;
}

void RasterLightingPass::renderGui(Gui::Window* pPassWindow)
{
    int dirty = 0;

    dirty |= (int)pPassWindow->checkbox("Debug Light Positions", mShowLights, false);
    if (mShowLights)
    {
        pPassWindow->text("     ");
        dirty |= (int)pPassWindow->var("Light Radius", mLightRadius, 0.01f, FLT_MAX, 0.1f, true);
    }
    dirty |= (int)pPassWindow->checkbox("Debug Cascade Distances", mShowCascades, false);

    pPassWindow->checkbox("Use Shadow Mapping", mUsingShadowMapping, false);
    // Shadow map properties
    if (mUsingShadowMapping)
    {
        // This has no effect for some reason
        pPassWindow->text("     ");
        if ((int)pPassWindow->checkbox("Front face culling", mCullFrontFace, true))
        {
            updateCullMode();
            dirty |= 1;
        }
        pPassWindow->text("     ");
        dirty |= (int)pPassWindow->checkbox("Use PCF Anti-Aliasing", mUsingPCF, true);
        pPassWindow->text("     Omnidirectional Shadow Properties");
        pPassWindow->text("     ");
        dirty |= (int)pPassWindow->var("Bias (pt)", mCubeShadowBias, 0.000001f, FLT_MAX, 0.000005f, true);
        pPassWindow->text("     ");
        int shadowChanged = (int)pPassWindow->var("zNear", mCubeShadowNear, 0.01f, FLT_MAX, 0.1f, true);
        pPassWindow->text("     ");
        shadowChanged |= (int)pPassWindow->var("zFar (pt)", mCubeShadowFar, 0.01f, FLT_MAX, 0.1f, true);
        // If near/far values are updated, recreate the shadow projection matrix
        if (shadowChanged)
        {
            mCubeShadowProj = glm::perspective(glm::radians(90.0f), 1.0f, mCubeShadowNear, mCubeShadowFar);
            dirty |= 1;
        }
        if (mUsingPCF)
        {
            pPassWindow->text("     ");
            dirty |= (int)pPassWindow->var("PCF Width (pt)", mCubePCFWidth, 0.0001f, FLT_MAX, 0.0001f, true);
        }

        pPassWindow->text("     Directional Shadow Properties");
        pPassWindow->text("     ");
        dirty |= (int)pPassWindow->var("Bias (dir)", mDirShadowBias, 0.000001f, FLT_MAX, 0.000005f, true);
        if (mUsingPCF)
        {
            pPassWindow->text("     ");
            dirty |= (int)pPassWindow->var("PCF Width (dir)", mDirPCFWidth, 0.0001f, FLT_MAX, 0.0001f, true);
        }
        pPassWindow->text("     ");
        dirty |= (int)pPassWindow->var("Ambient", mDirAmbient, 0.0f, FLT_MAX, 0.001f, true);
        pPassWindow->tooltip("To prevent the scene from being completely dark, we can let the "
            "shadowed region not be completely dark. This is the amount of light hitting shadowed regions.");
        pPassWindow->text("     Cascade limits");
        for (int i = 0; i < NUM_CASCADES; i++)
        {
            std::string label = "zFar [" + to_string(i) + "]";
            pPassWindow->text("     ");
            dirty |= (int)pPassWindow->var(label.c_str(), mDirShadowFar[i], 0.01f, FLT_MAX, 0.1f, true);
        }
    }

    // If any of our UI parameters changed, let the pipeline know we're doing something different next frame
    if (dirty) setRefreshFlag();
}

void RasterLightingPass::updateCullMode()
{
    RasterizerState::Desc rsDesc;
    rsDesc.setCullMode(mCullFrontFace ? RasterizerState::CullMode::Front : RasterizerState::CullMode::Back);
    mpCubeGfxState->setRasterizerState(RasterizerState::create(rsDesc));
    mpDirGfxState->setRasterizerState(RasterizerState::create(rsDesc));
}

// Based on CSM.cpp
void RasterLightingPass::camClipSpaceToWorldSpace(const Camera* pCamera, float3& center, float& radius, uint cascadeNum)
{
    float fovY = pCamera->getData().focalLength == 0.0f ? 0.0f : focalLengthToFovY(pCamera->getData().focalLength, pCamera->getData().frameHeight);
    float fovX = pCamera->getAspectRatio() * fovY;
    float n = cascadeNum == 0 ? pCamera->getNearPlane() : mDirShadowFar[cascadeNum - 1];
    float f = mDirShadowFar[cascadeNum]; // We use this value to limit the frustum size, otherwise the shadow map is too big

    glm::mat4 projMat = glm::perspective(fovY, pCamera->getAspectRatio(), n, f);
    glm::mat4 viewMat = pCamera->getViewMatrix();
    glm::mat4 viewProjMat = projMat * viewMat;
    glm::mat4 invViewProj = glm::inverse(viewProjMat);

    float3 clipSpace[8] =
    {
        float3(-1.0f, 1.0f, 0),
        float3(1.0f, 1.0f, 0),
        float3(1.0f, -1.0f, 0),
        float3(-1.0f, -1.0f, 0),
        float3(-1.0f, 1.0f, 1.0f),
        float3(1.0f, 1.0f, 1.0f),
        float3(1.0f, -1.0f, 1.0f),
        float3(-1.0f, -1.0f, 1.0f),
    };
    float3 viewFrustum[8] = {};

    // Get the center of the view frustum in world space
    //glm::mat4 invViewProj = pCamera->getInvViewProjMatrix();
    center = float3(0, 0, 0);
    for (uint32_t i = 0; i < 8; i++)
    {
        float4 crd = invViewProj * float4(clipSpace[i], 1.f);
        viewFrustum[i] = float3(crd) / crd.w;
        center += viewFrustum[i];
    }
    center *= (1.0f / 8.0f);

    // Calculate bounding sphere radius
    radius = 0;
    for (uint32_t i = 0; i < 8; i++)
    {
        float d = glm::length(center - viewFrustum[i]);
        radius = std::max(d, radius);
    }
}
