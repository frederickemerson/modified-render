/**********************************************************************************************************************
# Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without modification, are permitted provided that the
# following conditions are met:
#  * Redistributions of code must retain the copyright notice, this list of conditions and the following disclaimer.
#  * Neither the name of NVIDIA CORPORATION nor the names of its contributors may be used to endorse or promote products
#    derived from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.  IN NO EVENT
# SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA,
# OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
# LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
# ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
**********************************************************************************************************************/

#pragma once
#include "../DxrTutorSharedUtils/RenderPass.h"
#include "../DxrTutorSharedUtils/RayLaunch.h"

/**
 * Combines the visibility bitmap and the GBuffer for the final image.
 */
class VShadingPass : public ::RenderPass
{
public:
    using SharedPtr = std::shared_ptr<VShadingPass>;
    using SharedConstPtr = std::shared_ptr<const VShadingPass>;

    static SharedPtr create(const std::string& outBuf = ResourceManager::kOutputChannel, bool isHybridRendering = true) { return SharedPtr(new VShadingPass(outBuf, isHybridRendering)); }
    virtual ~VShadingPass() = default;

protected:
    VShadingPass(const std::string& outBuf, bool isHybridRendering) : ::RenderPass("Visibility-Shading Pass", "VShading Options") { 
        mOutputTexName = outBuf;
        mHybridMode = isHybridRendering;
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
    int32_t                                 mOutputIndex;           ///< An index for our output buffer
    std::string                             mOutputTexName;         ///< Where do we want to store the results?
    bool                                    mSkipShadows = false;   ///< Should we skip shadow computation?
    bool                                    mSkipAO = false;        ///< Should we skip ambient occlusion?

    bool                                    mDecodeMode = false;    ///< Do we perform shading, or just debug the visibility bitmap?
    int32_t                                 mDecodeBit = 0;         ///< If we are debugging visibility bitmap, which light should we see?
    bool                                    mDecodeVis = true;      ///< True for Visibility, false for AO

    float                                   mAmbient = 0.5f;        ///< Scene-dependent variable to avoid perfectly dark shadows
    int32_t                                 mNumAORays = 32;        ///< Number of AO rays shot per pixel

    bool                                    mHybridMode = true;     ///< True if doing hybrid rendering, else remote rendering.
};
