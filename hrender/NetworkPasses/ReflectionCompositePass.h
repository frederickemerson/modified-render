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
#include "Falcor.h"
#include "../DxrTutorSharedUtils/RenderPass.h"
#include "../DxrTutorSharedUtils/FullscreenLaunch.h"

/**
 * Combines the visibility bitmap and the GBuffer for the final image.
 */
class ReflectionCompositePass : public ::RenderPass
{
public:
    using SharedPtr = std::shared_ptr<ReflectionCompositePass>;
    using SharedConstPtr = std::shared_ptr<const ReflectionCompositePass>;

    static SharedPtr create() { return SharedPtr(new ReflectionCompositePass()); }
    virtual ~ReflectionCompositePass() = default;

protected:
    ReflectionCompositePass() : ::RenderPass("Reflection Composite Pass", "Reflection Composite Options") { }

    // Implementation of RenderPass interface
    bool initialize(RenderContext* pRenderContext, ResourceManager::SharedPtr pResManager) override;
    void initScene(RenderContext* pRenderContext, Scene::SharedPtr pScene) override;
    void execute(RenderContext* pRenderContext) override;
    void renderGui(Gui::Window* pPassWindow) override;

    // Override some functions that provide information to the RenderPipeline class
    bool requiresScene() override { return true; }
    bool appliesPostprocess() override { return true; }

    // Rendering state
    GraphicsState::SharedPtr                mpGfxState;
    Scene::SharedPtr                        mpScene;                ///< Our scene file (passed in from app)
    bool                                    mSkipRC = false;       ///< Should we skip SSR?

    // Shaders
    FullscreenLaunch::SharedPtr             mpRCShader;

    // Various internal parameters
    Fbo::SharedPtr							mpRCFbo;

    int                                     mTexWidth = -1;         ///< The width of the texture we render, based on the client
    int                                     mTexHeight = -1;        ///< The height of the texture we render, based on the client
};
