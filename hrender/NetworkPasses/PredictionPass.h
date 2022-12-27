
#ifndef PREDICTION_PASS_H
#define PREDICTION_PASS_H

#include "../DxrTutorSharedUtils/RenderPass.h"
#include "../DxrTutorSharedUtils/FullscreenLaunch.h"
#include "../DxrTutorSharedUtils/CircularBuffer.h"

/**
 * Based on scene information, output a motion vector that
 * predicts the motion of each pixel in the scene
 * between the old frame whose data we just received
 * from the server, and the current frame being rendered
 * by rasterisation on the client.
 *
 * The motion vectors are predicted using the scene's
 * information about its objects and their positions.
 *
 * This information is stored in a circular buffer.
 */
class PredictionPass : public ::RenderPass
{
public:
    using SharedPtr = std::shared_ptr<PredictionPass>;
    using SharedConstPtr = std::shared_ptr<const PredictionPass>;

    static SharedPtr create(int texWidth = -1, int texHeight = -1)
    {
        return SharedPtr(new PredictionPass(texWidth, texHeight));
    }

    virtual ~PredictionPass() = default;

protected:
    PredictionPass(int texWidth = -1, int texHeight = -1);

    // Implementation of RenderPass interface
    bool initialize(RenderContext* pRenderContext, ResourceManager::SharedPtr pResManager) override;
    void initScene(RenderContext* pRenderContext, Scene::SharedPtr pScene) override;
    void execute(RenderContext* pRenderContext) override;
    void renderGui(Gui::Window* pPassWindow) override;

    // Override some functions that provide information to the RenderPipeline class
    bool requiresScene() override { return false; }
    bool usesRasterization() override { return true; }
    bool usesRayTracing() override { return false; }

    // Rendering state
    Scene::SharedPtr                        mpScene;                        ///< Our scene file (passed in from app)
    FullscreenLaunch::SharedPtr             mpPredictShader;                ///< Compiled fragment shader for prediction
    FullscreenLaunch::SharedPtr             mpOffsetShader;                 ///< Compiled fragment shader for offsetting buffer
    FullscreenLaunch::SharedPtr             mpCopyShader;                   ///< Compiled fragment shader for copying buffer

    // Various internal parameters
    int32_t                                 mMotionVecIndex;                ///< An index for our output buffer
    int32_t                                 mVisBufOrigIndex;               ///< An index for the original visibility buffer
    int32_t                                 mVisBufOffsetIndex;             ///< An index for the offset visibility buffer
    int32_t                                 mAOBufOrigIndex;               ///< An index for the original AO buffer
    int32_t                                 mAOBufOffsetIndex;             ///< An index for the offset AO buffer
    int32_t                                 mRefBufOrigIndex;              ///< An index for the original RayTracing Reflection buffer
    int32_t                                 mRefBufOffsetIndex;             ///< An index for the offset RayTracing Reflection buffer
    int32_t                                 mWorldPosIndex;                 ///< An index for the world position
    int32_t                                 mZBufferIndex;                  ///< An index for the Z buffer
    bool                                    mUsePrediction = true;          ///< A flag to toggle prediction
    float                                   mOffsetFactor = 1.0f;           ///< The motion vector will be
                                                                            ///< multiplied by this value
    bool                                    mFirstGuiRender = true;         ///< A flag to set default GUI size
    int                                     mTexWidth = -1;                 ///< The width of the texture we render,
                                                                            ///  based on the client and macroblock size
    bool                                    mUsePercvDelay = true;          ///< A flag to toggle use of the delay that
                                                                            ///  is calculated by NetworkManager
    int                                     mPercvDelay = 0;
    int                                     mActualDelay = 10;
    int                                     mUnknownFragmentsMode = 0;
    int                                     mTexHeight = -1;                ///< The height of the texture we render,
                                                                            ///  based on the client and macroblock size
    CircularBuffer<CameraData>              camDataBuffer;                  ///< A buffer for camera data to be used
                                                                            ///  in prediction
};

#endif PREDICTION_PASS_H
