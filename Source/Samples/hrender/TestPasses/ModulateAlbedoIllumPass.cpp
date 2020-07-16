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

#include "ModulateAlbedoIllumPass.h"

namespace {
    // Where is our shader located?
    const char *kModulateShader        = "Samples\\hrender\\SVGFPasses\\Data\\SVGFPasses\\SVGFModulate.ps.hlsl";

    // Where the albedo values are output
    const char* kDirectAlbedoChannel   = "OutDirectAlbedo";
    const char* kIndirectAlbedoChannel = "OutIndirectAlbedo";
};

bool ModulateAlbedoIllumPass::initialize(RenderContext* pRenderContext, ResourceManager::SharedPtr pResManager)
{
    if (!pResManager) return false;

    // Stash a copy of our resource manager, allowing us to access shared rendering resources
    //    We need an output buffer; tell our resource manager we expect the standard output channel
    mpResManager = pResManager;
    mpResManager->requestTextureResource(mOutputChannel);
    mpResManager->requestTextureResource(mDirectIllumChannel);
    mpResManager->requestTextureResource(mIndirectIllumChannel);
    mpResManager->requestTextureResource(kDirectAlbedoChannel);
    mpResManager->requestTextureResource(kIndirectAlbedoChannel);

    // Create our graphics state and accumulation shader
    mpModulatePass = FullscreenLaunch::create(kModulateShader);

    return true;
}

void ModulateAlbedoIllumPass::resize(uint32_t width, uint32_t height)
{
    // We need a framebuffer to render to
    mpInternalFbo = mpResManager->createManagedFbo({ mOutputChannel });
}

void ModulateAlbedoIllumPass::renderGui(Gui::Window* pPassWindow)
{
    // Nothing in the GUI for now
}

void ModulateAlbedoIllumPass::execute(RenderContext* pRenderContext)
{
    // No valid framebuffer?  We're done.
    if (!mpInternalFbo) return;

    // Set shader parameters.
    auto shaderVars = mpModulatePass->getVars();
    shaderVars["gDirect"]      = mpResManager->getTexture(mDirectIllumChannel);
    shaderVars["gIndirect"]    = mpResManager->getTexture(mIndirectIllumChannel);
    shaderVars["gDirAlbedo"]   = mpResManager->getTexture(kDirectAlbedoChannel);
    shaderVars["gIndirAlbedo"] = mpResManager->getTexture(kIndirectAlbedoChannel);

    // Run the modulation pass
    mpModulatePass->execute(pRenderContext, mpInternalFbo);
}
