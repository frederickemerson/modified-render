#pragma once
#include "../DxrTutorSharedUtils/RasterLaunch.h"
#include "../DxrTutorSharedUtils/RayLaunch.h"
#include "../DxrTutorSharedUtils/RenderPass.h"

/** Ray traced ambient occlusion pass.
*/
class RasterLightingPass : public ::RenderPass, inherit_shared_from_this<::RenderPass, RasterLightingPass>
{
public:
    using SharedPtr = std::shared_ptr<RasterLightingPass>;
    using SharedConstPtr = std::shared_ptr<const RasterLightingPass>;

    static SharedPtr create(const std::string& outBuf = ResourceManager::kOutputChannel) { return SharedPtr(new RasterLightingPass(outBuf)); }
    virtual ~RasterLightingPass() = default;

protected:
    RasterLightingPass(const std::string& outBuf) : ::RenderPass("Rasterized Lighting", "Rasterized Lighting Options") { mOutputTexName = outBuf; }

    // Implementation of RenderPass interface
    bool initialize(RenderContext* pRenderContext, ResourceManager::SharedPtr pResManager) override;
    void initScene(RenderContext* pRenderContext, Scene::SharedPtr pScene) override;
    void execute(RenderContext* pRenderContext) override;
    void fillShadowMap(Falcor::RenderContext* pRenderContext, Light::SharedPtr currLight);
    void drawDebugLights();
    void renderGui(Gui::Window* pPassWindow) override;

    // Override some functions that provide information to the RenderPipeline class
    bool requiresScene() override { return true; }
    bool usesRayTracing() override { return true; }

    // Rendering state
    RayLaunch::SharedPtr                    mpRays;                 ///< Our wrapper around a DX Raytracing pass
    GraphicsState::SharedPtr                mpGfxState;             ///< Our graphics pipeline state
    RasterLaunch::SharedPtr                 mpShadowPass;           ///< Our raster pass used to create a shadow map 
    Scene::SharedPtr                        mpScene;                ///< Our scene file (passed in from app)  

    // Debug light drawing parameters/variables
    bool                                    mShowLights = true;     ///< Flag to render lights for debugging
    float                                   mLightRadius = 5.0f;    ///< The screen-space radius of the lights
    Buffer::SharedPtr                       mLightPosBuffer;        ///< Stores the positions of our lights

    // Various internal parameters
    int32_t                                 mOutputIndex;           ///< An index for our output buffer
    std::string                             mOutputTexName;         ///< Where do we want to store the results?

    // Shadow map parameters
    Texture::SharedPtr                      mpShadowMapTex;         ///< The z-values of the scene from the light's POV is stored here
    Texture::SharedPtr                      mpShadowMapDepthTex;    ///< Used for z-buffer of the shadow map rendering
    glm::mat4                               mShadowProj;            ///< Projection matrix of the shadow map (6 sides use the same projection matrix)
    uint32_t                                mShadowMapRes = 1024;   ///< Width/Height of the shadow map cube edge (square)
    float                                   mShadowNear = 0.5f;     ///< Near plane for shadow map
    float                                   mShadowFar = 15.0f;     ///< Far plane for the shadow map
    float                                   mShadowOffset = 1e-4f;  ///< Offset for shadow map to prevent shadow acnes
};
