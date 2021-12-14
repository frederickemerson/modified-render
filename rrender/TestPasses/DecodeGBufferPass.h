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
#include "../DxrTutorSharedUtils/FullscreenLaunch.h"
#include "GBufferComponents.slang"

class DecodeGBufferPass : public ::RenderPass
{
public:
    using SharedPtr = std::shared_ptr<DecodeGBufferPass>;
    using SharedConstPtr = std::shared_ptr<const DecodeGBufferPass>;

    static SharedPtr create(const std::string & outputBuffer) { return SharedPtr(new DecodeGBufferPass(outputBuffer)); }
    virtual ~DecodeGBufferPass() = default;

protected:
    DecodeGBufferPass(const std::string& outputBuffer) : mOutputChannel(outputBuffer), ::RenderPass("Decode GBuffer Pass", "Decode GBuffer Options") {}

    // Implementation of SimpleRenderPass interface
    bool initialize(RenderContext* pRenderContext, ResourceManager::SharedPtr pResManager) override;
    void execute(RenderContext* pRenderContext) override;
    void renderGui(Gui::Window* pPassWindow) override;
    void resize(uint32_t width, uint32_t height) override;

    // Override some functions that provide information to the RenderPipeline class
    bool appliesPostprocess() override { return true; }
    bool hasAnimation() override { return false; }

    // Information about the rendering texture we're accumulating into
    std::string                   mOutputChannel;

    // State for our decoder shader
    FullscreenLaunch::SharedPtr   mpDecodeShader;
    Fbo::SharedPtr                mpInternalFbo;

private:
    // Dropdown to select the buffer to decode
    Gui::DropdownList mGBufDropdown = { // Options displayed by the dropdown
        { 0, "WorldPosition" }, { 1, "WorldNormal" }, { 2, "MatlDiffuse" }, { 3, "MatlSpecular" },
        { 4, "MatlEmissive" }, { 5, "MatlDoubleSided" }, { 6, "MatlExtra" }
    };
    uint32_t mGBufComponentSelection = GBufComponent::MatlDiffuse; // Which buffer we should decode
};
