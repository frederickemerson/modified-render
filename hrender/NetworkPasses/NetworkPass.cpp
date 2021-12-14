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

#include "NetworkPass.h"

//std::vector<uint8_t> NetworkPass::normData = std::vector<uint8_t>();
std::vector<uint8_t> NetworkPass::posData = std::vector<uint8_t>(POS_TEX_LEN, 0);
int NetworkPass::posTexWidth = 0;
int NetworkPass::posTexHeight = 0;

//std::vector<uint8_t> NetworkPass::gBufData = std::vector<uint8_t>();
char* NetworkPass::visibilityDataForReadingClient = new char[VIS_TEX_LEN];
char* NetworkPass::visibilityDataForWritingClient = new char[VIS_TEX_LEN];
// for server side GPU-CPU trsf of visibilityBuffer, stores location of data, changes every frame
uint8_t* NetworkPass::pVisibilityDataServer = nullptr;
std::array<float3, 3> NetworkPass::camData;

bool NetworkPass::initialize(RenderContext* pRenderContext, ResourceManager::SharedPtr pResManager)
{
    // Stash a copy of our resource manager so we can get rendering resources
    mpResManager = pResManager;

    // Our GUI needs less space than other passes, so shrink the GUI window.
    setGuiSize(int2(300, 70));

    // Note that we some buffers from the G-buffer, plus the standard output buffer
    mpResManager->requestTextureResource("WorldPosition", ResourceFormat::RGBA32Float, ResourceManager::kDefaultFlags); // Only for client
    // For server buffers, we are creating them here, so we specify their width/height accordingly
    mpResManager->requestTextureResource("WorldPosition2", ResourceFormat::RGBA32Float, ResourceManager::kDefaultFlags);
    mpResManager->requestTextureResource("VisibilityBitmap", ResourceFormat::R32Uint, ResourceManager::kDefaultFlags);

    // Now that we've passed all our shaders in, compile and (if available) setup the scene
    if (mpScene) {
        mpRays->setScene(mpScene);
    }

    return true;
}

void NetworkPass::initScene(RenderContext* pRenderContext, Scene::SharedPtr pScene)
{
    // Stash a copy of the scene and pass it to our ray tracer (if initialized)
    mpScene = pScene;
    if (!mpScene) return;
    if (mpRays) {
        mpRays->setScene(mpScene);
    }
}

void NetworkPass::renderGui(Gui::Window* pPassWindow)
{
    int dirty = 0;
    pPassWindow->text(mMode == Mode::ServerRecv ? "Server receiver"
        : mMode == Mode::ServerSend ? "Server sender"
        : "Client");

    // If any of our UI parameters changed, let the pipeline know we're doing something different next frame
    if (dirty) setRefreshFlag();
}
