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

// This render pass is similar to the "SimpleDiffuseGIPass" from Tutorial #12, except it 
//     uses a more complex GGX illumination model (instead of a simple Lambertian model).
//     The changes mostly affect the HLSL shader code, as the C++ is quite similar

#pragma once
#include "../DxrTutorSharedUtils/RenderPass.h"
#include "../DxrTutorSharedUtils/RayLaunch.h"

class GGXClientGlobalIllumPass : public ::RenderPass
{
public:
    using SharedPtr = std::shared_ptr<GGXClientGlobalIllumPass>;
    using SharedConstPtr = std::shared_ptr<const GGXClientGlobalIllumPass>;

    static SharedPtr create(const std::string& outDirectIllum, const std::string& outIndirectIllum, int texWidth = -1, int texHeight = -1) {
        return SharedPtr(new GGXClientGlobalIllumPass(outDirectIllum, outIndirectIllum, texWidth, texHeight));
    }
    virtual ~GGXClientGlobalIllumPass() = default;

protected:
    GGXClientGlobalIllumPass(const std::string& outDirectIllum, const std::string& outIndirectIllum, int texWidth = -1, int texHeight = -1) :
        ::RenderPass("Client Global Illum., GGX BRDF", "GGX Global Illumination Options") {
        mDirectIllumTex = outDirectIllum; 
        mIndirectIllumTex = outIndirectIllum;
        mTexWidth = texWidth;
        mTexHeight = texHeight;
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
    RayLaunch::SharedPtr    mpRays;                       ///< Our wrapper around a DX Raytracing pass
    Scene::SharedPtr        mpScene;                      ///< Our scene file (passed in from app)  

    // Recursive ray tracing can be slow.  Add a toggle to disable, to allow you to manipulate the scene
    bool                    mDoIndirectGI = true;
    bool                    mDoDirectGI = true;

    //int32_t                 mUserSpecifiedRayDepth = 1;   ///<  What is the current maximum ray depth
    //const int32_t           mMaxPossibleRayDepth = 8;     ///<  The largest ray depth we support (without recompile)


    // What texture should was ask the resource manager to store our result in?
    std::string             mDirectIllumTex;
    std::string             mIndirectIllumTex;
    
    // Various internal parameters
    uint32_t                mFrameCount = 0x8465u;        ///< A frame counter to vary random numbers over time
    int                     mTexWidth = -1;               ///< The width of the texture we render, based on the client
    int                     mTexHeight = -1;              ///< The height of the texture we render, based on the client
};
