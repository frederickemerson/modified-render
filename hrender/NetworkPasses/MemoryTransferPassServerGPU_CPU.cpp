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

#include "MemoryTransferPassServerGPU_CPU.h"

bool MemoryTransferPassServerGPU_CPU::initialize(RenderContext* pRenderContext, ResourceManager::SharedPtr pResManager)
{
    // Stash a copy of our resource manager so we can get rendering resources
    mpResManager = pResManager;

    // Our GUI needs less space than other passes, so shrink the GUI window.
    setGuiSize(int2(300, 70));

    // Note that we some buffers from the G-buffer, plus the standard output buffer
    mpResManager->requestTextureResource("WorldPosition");

    // store the texture we will be transferring from
    visTex = mpResManager->getTexture("VisibilityBitmap");

    return true;
}

void MemoryTransferPassServerGPU_CPU::initScene(RenderContext* pRenderContext, Scene::SharedPtr pScene)
{
    // Stash a copy of the scene and pass it to our ray tracer (if initialized)
    mpScene = pScene;
    if (!mpScene) return;
}

void MemoryTransferPassServerGPU_CPU::execute(RenderContext* pRenderContext)
{
    // Load visibility texture from GPU to CPU

    // OLD METHOD: use if bugs start appearing
    //NetworkPass::visibilityData = visTex->getTextureData(pRenderContext, 0, 0, &NetworkPass::visibilityData);

    // New optimised method: old getTextureData() opens a buffer to the texture and copies data into our desired location
    // new getTextureData2() returns address of the buffer so we skip the copying to our desired location.
    // as a result, the location of this data (the ptr) changes with each call to getTextureData2;
    NetworkPass::pVisibilityData = visTex->getTextureData2(pRenderContext, 0, 0);
    OutputDebugString(L"\n\n= MemoryTransferPass - VisTex loaded to CPU =========");
}

void MemoryTransferPassServerGPU_CPU::renderGui(Gui::Window* pPassWindow)
{
    int dirty = 0;

    // If any of our UI parameters changed, let the pipeline know we're doing something different next frame
    if (dirty) setRefreshFlag();
}

std::vector<uint8_t> MemoryTransferPassServerGPU_CPU::texData(RenderContext* pRenderContext, Texture::SharedPtr tex)
{
    return tex->getTextureData(pRenderContext, 0, 0);
}