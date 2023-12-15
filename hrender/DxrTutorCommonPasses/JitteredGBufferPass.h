#pragma once
#include "../DxrTutorSharedUtils/FullscreenLaunch.h"
#include "../DxrTutorSharedUtils/RasterLaunch.h"
#include "../DxrTutorSharedUtils/RenderPass.h"
#include "../NetworkPasses/PredictionPass.h"

/** Rasterized GBuffer pass.
*       This GBuffer pass provides provides the option to render an environment map, and to use camera jitter for
*       anti-aliasing. The user can choose to jitter the camera with either the 8x MSAA pattern or randomized jitter.
*/
class JitteredGBufferPass : public ::RenderPass
{
public:
    using SharedPtr = std::shared_ptr<JitteredGBufferPass>;

    static SharedPtr create(int texWidth = -1, int texHeight = -1) { return SharedPtr(new JitteredGBufferPass(texWidth, texHeight)); }
    virtual ~JitteredGBufferPass() = default;

protected:
    JitteredGBufferPass(int texWidth = -1, int texHeight = -1) : ::RenderPass("Jittered G-Buffer", "Jittered G-Buffer Options") {
        mTexWidth = texWidth; mTexHeight = texHeight;
    }

    // Implementation of RenderPass interface
    bool initialize(RenderContext* pRenderContext, ResourceManager::SharedPtr pResManager) override;
    void execute(RenderContext* pRenderContext) override;
    void renderGui(Gui::Window* pPassWindow) override;
    void initScene(RenderContext* pRenderContext, Scene::SharedPtr pScene) override;

    // The RenderPass class defines various methods we can override to specify this pass' properties. 
    bool requiresScene() override { return true; }
    bool usesRasterization() override { return true; }
    bool usesEnvironmentMap() override { return true; }

    // Internal pass state
    GraphicsState::SharedPtr    mpGfxState;             ///< Our graphics pipeline state (i.e., culling, raster, blend settings)
    Scene::SharedPtr            mpScene;                ///< A pointer to the scene we're rendering
    RasterLaunch::SharedPtr     mpRaster;               ///< A wrapper managing the shader for our g-buffer creation
    FullscreenLaunch::SharedPtr mpClearGBuf;            ///< A wrapper over the shader to clear our g-buffer to the env map
    bool                        mUseJitter = true;      ///< Jitter the camera?
    bool                        mUseRandom = false;     ///< If jittering, use random samples or 8x MSAA pattern?
    int                         mFrameCount = 0;        ///< If jittering the camera, which frame in our jitter are we on?
    bool                        mUseEnvMap = true;      ///< Using environment map?

    // Our random number generator (if we're doing randomized samples)
    std::uniform_real_distribution<float> mRngDist;     ///< We're going to want random #'s in [0...1] (the default distribution)
    std::mt19937 mRng;                                  ///< Our random number generator.  Set up in initialize()

    // What's our background color?
    float3  mBgColor = float3(0.5f, 0.5f, 1.0f);        ///<  Color stored into our diffuse G-buffer channel if we hit no geometry

    int                                     mTexWidth = -1;         ///< The width of the texture we render, based on the client
    int                                     mTexHeight = -1;        ///< The height of the texture we render, based on the client
};
