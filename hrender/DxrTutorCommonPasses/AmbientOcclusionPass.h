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

/** Ray traced ambient occlusion pass.
*/
class AmbientOcclusionPass : public ::RenderPass
{
public:
    using SharedPtr = std::shared_ptr<AmbientOcclusionPass>;
    using SharedConstPtr = std::shared_ptr<const AmbientOcclusionPass>;

    static SharedPtr create(const std::string &outBuf = ResourceManager::kOutputChannel, int texWidth = -1, int texHeight = -1) { 
        return SharedPtr(new AmbientOcclusionPass(outBuf, texWidth, texHeight)); }
    virtual ~AmbientOcclusionPass() = default;

protected:
    AmbientOcclusionPass(const std::string &outBuf, int texWidth = -1, int texHeight = -1) : ::RenderPass("Ambient Occlusion Pass", "Ambient Occlusion Options") {
        mOutputTexName = outBuf; mTexWidth = texWidth; mTexHeight = texHeight;
    }

    // Implementation of RenderPass interface
    bool initialize(RenderContext* pRenderContext, ResourceManager::SharedPtr pResManager) override;
    void initScene(RenderContext* pRenderContext, Scene::SharedPtr pScene) override;
    void renderGui(Gui::Window* pPassWindow) override;
    void execute(RenderContext* pRenderContext) override;

    // Override some functions that provide information to the RenderPipeline class
    bool requiresScene() override { return true; }
    bool usesRayTracing() override { return true; }

    // Rendering state
    RayLaunch::SharedPtr                    mpRays;                 ///< Our wrapper around a DX Raytracing pass
    Scene::SharedPtr                        mpScene;                ///< Our scene file (passed in from app)  
    
    // Various internal parameters
    bool                                    mSkipAo = false;        ///< Should we skip ambient occlusion?
    float                                   mAORadius = 0.0f;       ///< What radius are we using for AO rays (i.e., maxT when ray tracing)
    uint32_t                                mFrameCount = 0x4641u;  ///< Frame count used to help seed our shaders' random number generator
    int32_t                                 mNumRaysPerPixel = 32;  ///< How many ambient occlusion rays should we shot per pixel?
    int                                     mTexWidth = -1;         ///< The width of the texture we render, based on the client
    int                                     mTexHeight = -1;        ///< The height of the texture we render, based on the client

    // Indices we can use to query the resource manager for various texture resources
    int32_t                                 mPositionIndex;         ///< An index for the G-Buffer wsPosition buffer
    int32_t                                 mNormalIndex;           ///< An index for the G-Buffer wsNormal buffer
    int32_t                                 mOutputIndex;           ///< An index for our output buffer

    // The name of the buffer we want to store our computations into.
    std::string                             mOutputTexName;         ///< Where do we want to store the results?
};
