#pragma once
#include "../DxrTutorSharedUtils/RenderPass.h"
#include "../DxrTutorSharedUtils/RasterLaunch.h"
#include <random>

class JitteredGBufferPass : public ::RenderPass, inherit_shared_from_this<::RenderPass, JitteredGBufferPass> {
public:
    using SharedPtr = std::shared_ptr<JitteredGBufferPass>;

    static SharedPtr create() { return SharedPtr(new JitteredGBufferPass()); }
    virtual ~JitteredGBufferPass() = default;

protected:
    JitteredGBufferPass() : ::RenderPass("Jittered G-Buffer", "Jittered G-Buffer Options") {}

    // Implementation of RenderPass interface
    bool initialize(RenderContext* pRenderContext, ResourceManager::SharedPtr pResManager) override;
    void execute(RenderContext* pRenderContext, GraphicsState* pDefaultGfxState) override;
    void renderGui(Gui* pGui, Gui::Window* pPassWindow) override;
    void initScene(RenderContext* pRenderContext, Scene::SharedPtr pScene) override;

    // The RenderPass class defines various methods we can override to specify this pass' properties. 
    bool requiresScene() override { return true; }
    bool usesRasterization() override { return true; }

    // Internal pass state
    GraphicsState::SharedPtr    mpGfxState;             ///< Our graphics pipeline state (i.e., culling, raster, blend settings)
    Scene::SharedPtr            mpScene;                ///< A pointer to the scene we're rendering
    RasterLaunch::SharedPtr     mpRaster;               ///< A wrapper managing the shader for our g-buffer creation
    bool                        mUseJitter = true;      ///< Jitter the camera?
    bool                        mUseRandom = false;     ///< If jittering, use random samples or 8x MSAA pattern?
    int                         mFrameCount = 0;        ///< If jittering the camera, which frame in our jitter are we on?

    // Our random number generator (if we're doing randomized samples)
    std::uniform_real_distribution<float> mRngDist;     ///< We're going to want random #'s in [0...1] (the default distribution)
    std::mt19937 mRng;                                  ///< Our random number generator.  Set up in initialize()

    // What's our background color?
    float3  mBgColor = float3(0.5f, 0.5f, 1.0f);            ///<  Color stored into our diffuse G-buffer channel if we hit no geometry
};
