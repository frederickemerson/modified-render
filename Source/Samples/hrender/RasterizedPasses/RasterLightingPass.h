#pragma once
#include "../DxrTutorSharedUtils/RasterLaunch.h"
#include "../DxrTutorSharedUtils/RayLaunch.h"
#include "../DxrTutorSharedUtils/RenderPass.h"

// When updating this, mDirShadowFar must also be updated to include NUM_CASCADES entries.
// For drawing debug cascades, the rasterLighting.hlsl function cascadeColor() must also 
// be updated to have enough colors to shade each cascade differently.
#define NUM_CASCADES 3

/** Ray traced ambient occlusion pass.
*/
class RasterLightingPass : public ::RenderPass, inherit_shared_from_this<::RenderPass, RasterLightingPass>
{
public:
    using SharedPtr = std::shared_ptr<RasterLightingPass>;
    using SharedConstPtr = std::shared_ptr<const RasterLightingPass>;

    static SharedPtr create(const std::string& outBuf = ResourceManager::kOutputChannel) { return SharedPtr(new RasterLightingPass(outBuf)); }
    virtual ~RasterLightingPass() = default;

    // Used to pass frustum bounds
    struct Bounds
    {
        float3 min = float3(std::numeric_limits<float>::max());
        float3 max = float3(std::numeric_limits<float>::min());
    };

protected:
    RasterLightingPass(const std::string& outBuf) : ::RenderPass("Rasterized Lighting", "Rasterized Lighting Options") { mOutputTexName = outBuf; }

    // Implementation of RenderPass interface
    bool initialize(RenderContext* pRenderContext, ResourceManager::SharedPtr pResManager) override;
    void initScene(RenderContext* pRenderContext, Scene::SharedPtr pScene) override;
    void execute(RenderContext* pRenderContext) override;
    void renderGui(Gui::Window* pPassWindow) override;

    // Render the scene from the perspective of the specified point light
    void fillShadowCubeMap(Falcor::RenderContext* pRenderContext, Light::SharedPtr currLight, uint pointLightIdx);
    // Render the scene from the perspective of the specified directional light
    void fillShadowDirMap(Falcor::RenderContext* pRenderContext, Light::SharedPtr currLight);
    // Compute the world coordinates of the provided camera's view frustum
    void camClipSpaceToWorldSpace(const Camera* pCamera, float3& center, float& radius, uint cascadeNum);
    // Send light information to the lighting shader required to draw the light's locations
    void drawDebugLights();
    // Update the cubemap's graphics state to enable/disable front face culling.
    // This is the canonical way to do it (see Falcor GBufferRaster), but does not seem to work.
    void updateCullMode();

    // Override some functions that provide information to the RenderPipeline class
    bool requiresScene() override { return true; }
    bool usesRayTracing() override { return true; }

    // Rendering state
    RayLaunch::SharedPtr            mpRays;                 ///< Our wrapper around a DX Raytracing pass
    GraphicsState::SharedPtr        mpCubeGfxState;         ///< Our graphics pipeline state for the cubemap shadow pass
    GraphicsState::SharedPtr        mpDirGfxState;          ///< Our graphics pipeline state for the directional shadow pass
    RasterLaunch::SharedPtr         mpShadowPass;           ///< Our raster pass used to create a directional shadow map 
    RasterLaunch::SharedPtr         mpShadowCubePass;       ///< Our raster pass used to create an omnidirectional shadow map 
    Scene::SharedPtr                mpScene;                ///< Our scene file (passed in from app)  

    // Debug light drawing/cascade drawing parameters/variables
    bool                            mShowLights = true;     ///< Flag to render lights for debugging
    float                           mLightRadius = 5.0f;    ///< The screen-space radius of the lights
    Buffer::SharedPtr               mLightPosBuffer;        ///< Stores the positions of our lights
    bool                            mShowCascades = false;  ///< Flag to render cascades (of dir shadow map) for debugging

    // Various internal parameters
    int32_t                         mOutputIndex;           ///< An index for our output buffer
    std::string                     mOutputTexName;         ///< Where do we want to store the results?

    // Omnidirectional Shadow map parameters
    uint                            mNumPointLights;        ///< The total number of point lights. This informs how many shadow cubemaps we need
    uint                            mNumPointLightsVisible; ///< The total number of point lights in frame. This is for drawing of debug lights.
    Texture::SharedPtr              mpCubeShadowMapTex;     ///< The z-values of the scene from the point light's POV is stored here
    Texture::SharedPtr              mpCubeShadowMapZTex;    ///< Used for z-buffer of the shadow cube map rendering
    glm::mat4                       mCubeShadowProj;        ///< Projection matrix of the shadow map (6 sides use the same projection matrix)
    uint32_t                        mCubeShadowMapRes = 1024; ///< Width/Height of the shadow map cube edge (square)
    float                           mCubeShadowNear = 0.5f; ///< Near plane for cube shadow map
    float                           mCubeShadowFar = 15.0f; ///< Far plane for the cube shadow map
    float                           mCubeShadowBias = 1e-4f;///< Offset for omnidirectional shadow map to prevent shadow acnes

    uint                            mNumDirLights;          ///< The total number of directional lights. This informs whether we need an ortho shadow map
    Texture::SharedPtr              mpDirShadowMapTex;      ///< The z-values of the scene from the directional light's POV is stored here
    Texture::SharedPtr              mpDirShadowMapZTex;     ///< Used for z-buffer of the shadow map (directional) rendering
    uint32_t                        mDirShadowMapRes = 2048;///< Width/Height of the shadow map cube edge (square)
    float                           mDirShadowBias = 3e-3f; ///< Offset for directional shadow map to prevent shadow acnes
    float                           mDirAmbient = 0.0f;     ///< Percentage of light used even in shadow (prevent scene from being too dark)

    float                           mDirShadowFar[NUM_CASCADES] = {     ///< Far plane for the directional shadow map
                                        5.0f, 15.0f, 40.0f
    };
    float3                          mDirShadowOrigin[NUM_CASCADES];     ///< Origins of the directional shadow map's cascades
    Buffer::SharedPtr               mDirShadowOriginBuffer;             ///< The buffer used to transfer the origins of cascades to the GPU
    float                           mDirShadowRadius[NUM_CASCADES];     ///< Radius of view frustums for directional shadow map's cascades
    Buffer::SharedPtr               mDirShadowRadiusBuffer;             ///< The buffer used to transfer the radii of cascades to the GPU
    glm::mat4                       mDirShadowViewProj[NUM_CASCADES];   ///< The view-projection matrices of the directional shadow map's cascades
    Buffer::SharedPtr               mDirShadowViewProjBuffer;             ///< The buffer used to transfer the VP-matrices of cascades to the GPU

    // Shadow mapping options
    bool                            mUsingShadowMapping = true; ///< Whether we want to enable shadow mapping
    bool                            mCullFrontFace = true;  ///< True if we want to cull front faces
    bool                            mUsingPCF = true;       ///< True if we want to use PCF anti-aliasing
    float                           mCubePCFWidth = 0.007f; ///< The width of the PCF filter
    float                           mDirPCFWidth = 0.0005f; ///< The width of the PCF filter
};
