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

bool NetworkPass::initialize(RenderContext* pRenderContext, ResourceManager::SharedPtr pResManager)
{
    // Stash a copy of our resource manager so we can get rendering resources
    mpResManager = pResManager;

    // Our GUI needs less space than other passes, so shrink the GUI window.
    setGuiSize(int2(300, 70));

    // Note that we some buffers from the G-buffer, plus the standard output buffer
    mpResManager->requestTextureResource("WorldPosition");
    mpResManager->requestTextureResource("WorldNormal");
    mpResManager->requestTextureResource("__TextureData");
    mpResManager->requestTextureResource("WorldPosition2");
    mpResManager->requestTextureResource("WorldNormal2");
    mpResManager->requestTextureResource("__TextureData2");
    mOutputIndex = mpResManager->requestTextureResource(mOutputTexName, ResourceFormat::R32Uint);

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

void NetworkPass::execute(RenderContext* pRenderContext)
{
    if (mMode == Mode::Server)
        executeServerSend(pRenderContext);
    else if (mMode == Mode::ServerSend)
        executeServerRecv(pRenderContext);
    else 
        executeClient(pRenderContext);
}

std::vector<uint8_t> NetworkPass::texData(RenderContext* pRenderContext, Texture::SharedPtr tex)
{
    return tex->getTextureData(pRenderContext, 0, 0, "TestFile.png");
}


bool NetworkPass::firstClientRender(RenderContext* pRenderContext)
{
    // Send scene
    //Texture::SharedPtr posTex = mpResManager->getTexture("WorldPosition");
    //Texture::SharedPtr normTex = mpResManager->getTexture("WorldNormal");
    //Texture::SharedPtr gBufTex = mpResManager->getTexture("__TextureData");


    mFirstRender = false;
    //std::vector<uint8_t> posData = texData(pRenderContext, posTex);
    //std::vector<uint8_t> normData = texData(pRenderContext, normTex);
    //std::vector<uint8_t> gBufData = texData(pRenderContext, gBufTex);
    
    return true;
}

void NetworkPass::executeClient(RenderContext* pRenderContext)
{
    // Slight branch optimization over:
    // if (!mFirstRender) firstClientRender();
    !mFirstRender || firstClientRender(pRenderContext);

    Texture::SharedPtr normTex = mpResManager->getTexture("WorldNormal");
    Texture::SharedPtr normTex2 = mpResManager->getTexture("WorldNormal2");
    Texture::SharedPtr posTex = mpResManager->getTexture("WorldPosition");
    Texture::SharedPtr posTex2 = mpResManager->getTexture("WorldPosition2");
    Texture::SharedPtr gBufTex = mpResManager->getTexture("__TextureData");
    Texture::SharedPtr gBufTex2 = mpResManager->getTexture("__TextureData2");

    std::vector<uint8_t> normData = texData(pRenderContext, normTex);
    std::vector<uint8_t> posData = texData(pRenderContext, posTex);
    std::vector<uint8_t> gBufData = texData(pRenderContext, gBufTex);

    //Texture::SharedPtr pOther = Texture::create2D(getWidth(mipLevel), getHeight(mipLevel), ResourceFormat::RGBA32Float, 1, 1, nullptr, ResourceBindFlags::RenderTarget | ResourceBindFlags::ShaderResource);
    /*pRenderContext->blit(normTex->getSRV(0, 1, 0, 1), normTex2->getRTV(0, 0, 1));
    pRenderContext->blit(posTex->getSRV(0, 1, 0, 1), posTex2->getRTV(0, 0, 1));
    pRenderContext->blit(gBufTex->getSRV(0,  1, 0, 1), gBufTex2->getRTV(0, 0, 1));*/

    // Send the three textures to server

    // Await server to send back the visibility pass texture
}

bool NetworkPass::firstServerRender(RenderContext* pRenderContext)
{
    // Send scene
    mFirstRender = false;
    Texture::SharedPtr posTex = mpResManager->getTexture("WorldPosition");
    Texture::SharedPtr normTex = mpResManager->getTexture("WorldNormal");
    Texture::SharedPtr gBufTex = mpResManager->getTexture("__TextureData");
    return true;
}

void NetworkPass::executeServerRecv(RenderContext* pRenderContext)
{
    mFirstRender || firstServerRender(pRenderContext);
    
    //Texture::SharedPtr pTex = Texture::create2D(mpResManager->getWidth(), mpResManager->getHeight(), texFormat, 1, generateMipLevels ? Texture::kMaxPossible : 1, pBitmap->getData(), bindFlags);

    // Await the three textures from client
}

void NetworkPass::executeServerSend(RenderContext* pRenderContext)
{
    // Send back the visibility pass texture
}

void NetworkPass::renderGui(Gui::Window* pPassWindow)
{
    int dirty = 0;
    pPassWindow->text(mMode == Mode::Server ? "Server receiver"
        : mMode == Mode::ServerSend ? "Server sender"
        : "Client");

    // If any of our UI parameters changed, let the pipeline know we're doing something different next frame
    if (dirty) setRefreshFlag();
}

