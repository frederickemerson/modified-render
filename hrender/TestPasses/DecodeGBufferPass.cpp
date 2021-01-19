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

#include "DecodeGBufferPass.h"

namespace
{
    const char *kDecodeShader = "Samples\\hrender\\TestPasses\\Data\\TestPasses\\DecodeGBufferPass.ps.hlsl";
};

bool DecodeGBufferPass::initialize(RenderContext* pRenderContext, ResourceManager::SharedPtr pResManager)
{
    if (!pResManager) return false;

    // Stash our resource manager; ask for the texture the developer asked us to output to, as well as
    // the textures our GBuffers write to
    mpResManager = pResManager;
    mpResManager->requestTextureResources({ mOutputChannel, "WorldPosition", "WorldNormal", "__TextureData" });

    setGuiSize(int2(300, 70));

    // Create our graphics state and accumulation shader
    mpDecodeShader = FullscreenLaunch::create(kDecodeShader);

    return true;
}

void DecodeGBufferPass::resize(uint32_t width, uint32_t height)
{
    // We need a framebuffer to render to
    mpInternalFbo = mpResManager->createManagedFbo({ mOutputChannel });
}

void DecodeGBufferPass::renderGui(Gui::Window* pPassWindow)
{
    if (pPassWindow->dropdown("Displayed", mGBufDropdown, mGBufComponentSelection))
    {
        // If we modify the rendered pass, notify the pipeline
        setRefreshFlag();
    }
}

void DecodeGBufferPass::execute(RenderContext* pRenderContext)
{
    // If our input texture is invalid do nothing.
    if (!mpInternalFbo) return;
   
    // Set shader parameters for our accumulation
    auto shaderVars = mpDecodeShader->getVars();
    shaderVars["gWsPos"]    = mpResManager->getTexture("WorldPosition");
    shaderVars["gWsNorm"]   = mpResManager->getTexture("WorldNormal");
    shaderVars["gTexData"]  = mpResManager->getTexture("__TextureData");
    // TODO: There should be a better way to pass enums to shader, but this works for now
    shaderVars["DecoderCB"]["gBufComponent"] = mGBufComponentSelection;

    // Do the accumulation
    mpDecodeShader->execute(pRenderContext, mpInternalFbo);
}
