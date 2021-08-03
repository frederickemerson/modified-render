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
 * Perform ray tracing based on GBuffer content to produce
 * shadow/visibility bitmap.
*/
class VisibilityPass : public ::RenderPass
{
public:
    using SharedPtr = std::shared_ptr<VisibilityPass>;
    using SharedConstPtr = std::shared_ptr<const VisibilityPass>;

    static SharedPtr create(const std::string& outBuf = ResourceManager::kOutputChannel, const std::string& posBuf = "WorldPosition2", int texWidth = -1, int texHeight = -1) {
        return SharedPtr(new VisibilityPass(outBuf, posBuf, texWidth, texHeight));
    }
    virtual ~VisibilityPass() = default;

protected:
    VisibilityPass(const std::string& outBuf, const std::string& posBuf, int texWidth = -1, int texHeight = -1) : ::RenderPass("Visibility Pass", "Visibility Pass Options") {
        mOutputTexName = outBuf; mPosBufName = posBuf;  mTexWidth = texWidth; mTexHeight = texHeight;
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

    // compression
    std::vector<uint8_t>                    visibilityData;
    int                                     srcSize = 0;
    char*                                   srcData;
    char*                                   dstData;
    char*                                   srcData2;
    //void*                                   state; // for compression buffer

    // benchmarking
    int                                     frequency = 600;
    int                                     counter = 0;
    int                                     compressed_size = 0;
    std::chrono::microseconds::rep          gpucpu_duration = 0;
    std::chrono::microseconds::rep          compress_duration = 0;
    std::chrono::microseconds::rep          cpugpu_duration = 0;
    std::chrono::microseconds::rep          decompress_duration = 0;

    // Various internal parameters
    int32_t                                 mOutputIndex;           ///< An index for our output buffer
    std::string                             mOutputTexName;         ///< Where do we want to store the results?
    std::string                             mPosBufName = "WorldPosition";            ///< Where to find the position buffer
    bool                                    mSkipShadows = false;   ///< Should we skip shadow computation?
    int                                     mTexWidth = -1;         ///< The width of the texture we render, based on the client
    int                                     mTexHeight = -1;        ///< The height of the texture we render, based on the client
};
