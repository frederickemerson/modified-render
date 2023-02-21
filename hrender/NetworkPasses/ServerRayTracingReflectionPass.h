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
 * Calculate Server Raytracing reflections.
 */
class ServerRayTracingReflectionPass : public ::RenderPass
{
public:
    using SharedPtr = std::shared_ptr<ServerRayTracingReflectionPass>;
    using SharedConstPtr = std::shared_ptr<const ServerRayTracingReflectionPass>;

    static SharedPtr create(const std::string& outBuf = ResourceManager::kOutputChannel, int texWidth = -1, int texHeight = -1) { return SharedPtr(new ServerRayTracingReflectionPass(outBuf, texWidth, texHeight)); }
    virtual ~ServerRayTracingReflectionPass() = default;

protected:
    ServerRayTracingReflectionPass(const std::string& outBuf, int texWidth = -1, int texHeight = -1) : ::RenderPass("SRT Reflection Pass", "Ray Tracing Reflection Options") { mOutputTexName = outBuf; mTexWidth = texWidth; mTexHeight = texHeight;
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
    bool                                    mSkipSRT = false;       ///< Should we skip SSR?
    float                                   mRoughnessThreshold = 0.4f; ///< Controls how shiny required for reflections to work
    float                                   mLumThreshold = 0.15f;   ///< Controls how bright each reflection should be to be accepted
    bool                                    mUseThresholds = true; ///< Do we want to limit the colour output?
    // Various internal parameters
    int32_t                                 mOutputIndex;           ///< An index for our output buffer
    std::string                             mOutputTexName;         ///< Where do we want to store the results?
    int                                     mTexWidth = -1;         ///< The width of the texture we render, based on the client
    int                                     mTexHeight = -1;        ///< The height of the texture we render, based on the client
};
