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

// Simple full screen pass that combines the output of the ggxGlobalIlluminationDemod pass
//    to make the full GGX output.

#pragma once

// Include our base render pass
#include "../DxrTutorSharedUtils/RenderPass.h"

// Include the wrapper that makes launching full-screen raster work simple.
#include "../DxrTutorSharedUtils/FullscreenLaunch.h"

class ModulateAlbedoIllumPass : public ::RenderPass, inherit_shared_from_this<::RenderPass, ModulateAlbedoIllumPass>
{
public:
    using SharedPtr = std::shared_ptr<ModulateAlbedoIllumPass>;

    static SharedPtr create(const std::string& directIllum, const std::string& indirectIllum, const std::string& outputChannel) { return SharedPtr(new ModulateAlbedoIllumPass(directIllum, indirectIllum, outputChannel)); }
    virtual ~ModulateAlbedoIllumPass() = default;

protected:
    ModulateAlbedoIllumPass(const std::string& directIllum, const std::string& indirectIllum, const std::string& output):
        mDirectIllumChannel(directIllum), mIndirectIllumChannel(indirectIllum), mOutputChannel(output), 
        ::RenderPass("Modulate Albedo/Illum", "Modulate Options") {}

    // Implementation of RenderPass interface
    bool initialize(RenderContext* pRenderContext, ResourceManager::SharedPtr pResManager) override;
    void execute(RenderContext* pRenderContext) override;
    void renderGui(Gui::Window* pPassWindow) override;
    void resize(uint32_t width, uint32_t height) override;

    // The base RenderPass class defines a number of methods that we can override to specify
    //    what properties this pass has.
    bool appliesPostprocess() override { return true; }
    bool hasAnimation() override { return false; }      // Removes a GUI control that is confusing for this simple demo

    // Information about the rendering texture we output to
    std::string                   mOutputChannel;
    std::string                   mDirectIllumChannel;
    std::string                   mIndirectIllumChannel;

    // Internal pass state
    FullscreenLaunch::SharedPtr   mpModulatePass;         ///< Our modulation shader state
    Fbo::SharedPtr                mpInternalFbo;          ///< The FBO for our shader output
};
