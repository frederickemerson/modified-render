#pragma once
#include "../DxrTutorSharedUtils/RenderPass.h"
#include "../DxrTutorSharedUtils/RayLaunch.h"
#include "../DxrTutorSharedUtils/RenderConfig.h"
#include "../DxrTutorSharedUtils/FullscreenLaunch.h"
/**
 * Takes in a YUV444P texture and converts it to RGBA32Float format.
*/
class ClientRemoteConverter : public ::RenderPass
{
public:
    using SharedPtr = std::shared_ptr<ClientRemoteConverter>;
    using SharedConstPtr = std::shared_ptr<const ClientRemoteConverter>;

    static SharedPtr create(const std::string& inBuf, const std::string& outBuf) {
        return SharedPtr(new ClientRemoteConverter(inBuf, outBuf));
    }
    virtual ~ClientRemoteConverter() = default;

protected:
    ClientRemoteConverter(const std::string& inBuf, const std::string& outBuf, int texWidth = -1, int texHeight = -1) : ::RenderPass("YUV to RGBA Pass", "YUV to RGBA Pass Options") {
        mInputTexName = inBuf; mOutputTexName = outBuf;
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

    FullscreenLaunch::SharedPtr mpClearGBuf;            ///< A wrapper over the shader to clear our g-buffer to the env map
};
