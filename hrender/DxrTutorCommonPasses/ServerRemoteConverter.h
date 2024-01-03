#pragma once
#include "../DxrTutorSharedUtils/RenderPass.h"
#include "../DxrTutorSharedUtils/RayLaunch.h"
#include "../DxrTutorSharedUtils/RenderConfig.h"
#include "../DxrTutorSharedUtils/FullscreenLaunch.h"
/**
 * Takes in a RGBA32Float texture and converts it to R11G11B10 format.
*/
class ServerRemoteConverter : public ::RenderPass
{
public:
    using SharedPtr = std::shared_ptr<ServerRemoteConverter>;
    using SharedConstPtr = std::shared_ptr<const ServerRemoteConverter>;

    static SharedPtr create(const std::string& inBuf, const std::string& outBuf, int texWidth = -1, int texHeight = -1) {
        return SharedPtr(new ServerRemoteConverter(inBuf, outBuf, texWidth, texHeight));
    }
    virtual ~ServerRemoteConverter() = default;

protected:
    ServerRemoteConverter(const std::string& inBuf, const std::string& outBuf, int texWidth = -1, int texHeight = -1) : ::RenderPass("Float compaction pass", "Float compaction pass options") {
        mInputTexName = inBuf; mOutputTexName = outBuf;
        mTexWidth = texWidth; mTexHeight = texHeight;
    }

    // Implementation of RenderPass interface
    bool initialize(RenderContext* pRenderContext, ResourceManager::SharedPtr pResManager) override;
    void initScene(RenderContext* pRenderContext, Scene::SharedPtr pScene) override;
    void execute(RenderContext* pRenderContext) override;
    void renderGui(Gui::Window* pPassWindow) override;

    // Override some functions that provide information to the RenderPipeline class
    bool requiresScene() override { return true; }
    bool usesRayTracing() override { return true; }

    // Rendering state
    RayLaunch::SharedPtr                    mpRays;                 ///< Our wrapper around a DX Raytracing pass
    Scene::SharedPtr                        mpScene;                ///< Our scene file (passed in from app)

    // Various internal parameters
    int32_t                                 mInputIndex;           ///< An index for our output buffer
    std::string                             mInputTexName;          ///< Where are the data we are converting?
    int32_t                                 mOutputIndex;           ///< An index for our output buffer
    std::string                             mOutputTexName;         ///< Where do we want to store the results?
    int                                     mTexWidth = -1;         ///< The width of the texture we render, based on the client
    int                                     mTexHeight = -1;        ///< The height of the texture we render, based on the client
    FullscreenLaunch::SharedPtr mpClearGBuf;            ///< A wrapper over the shader to clear our g-buffer to the env map
};
