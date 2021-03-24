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
std::vector<uint8_t> NetworkPass::visibilityData = std::vector<uint8_t>(VIS_TEX_LEN, 0);
std::array<float3, 3> NetworkPass::camData;

bool NetworkPass::initialize(RenderContext* pRenderContext, ResourceManager::SharedPtr pResManager)
{
    // Stash a copy of our resource manager so we can get rendering resources
    mpResManager = pResManager;

    // Our GUI needs less space than other passes, so shrink the GUI window.
    setGuiSize(int2(300, 70));

    // Note that we some buffers from the G-buffer, plus the standard output buffer
    mpResManager->requestTextureResource("WorldPosition"); // Only for client
    // For server buffers, we are creating them here, so we specify their width/height accordingly
    mpResManager->requestTextureResource("WorldPosition2", ResourceFormat::RGBA16Float, ResourceManager::kDefaultFlags, mTexWidth, mTexHeight);
    mpResManager->requestTextureResource("VisibilityBitmap", ResourceFormat::R32Uint, ResourceManager::kDefaultFlags, mTexWidth, mTexHeight);

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
    if (mMode == Mode::ServerSend)
        executeServerSend(pRenderContext);
    else if (mMode == Mode::Server)
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
    NetworkManager::SharedPtr pNetworkManager = mpResManager->mNetworkManager;
    // Send the texture size to the server
    OutputDebugString(L"\n\n= Awaiting width/height sending over network... =========\n\n");
    pNetworkManager->SendInt(mpResManager->getWidth(), pNetworkManager->mConnectSocket);
    pNetworkManager->SendInt(mpResManager->getHeight(), pNetworkManager->mConnectSocket);
    OutputDebugString(L"\n\n= width/height sent over network =========\n\n");

    // TODO: Send scene

    mFirstRender = false;
    
    return true;
}

void NetworkPass::executeClient(RenderContext* pRenderContext)
{
    NetworkManager::SharedPtr pNetworkManager = mpResManager->mNetworkManager;

    // Slight branch optimization over:
     //if (!mFirstRender) firstClientRender();
    mFirstRender && firstClientRender(pRenderContext);

    // Send camera data from client to server
    Camera::SharedPtr cam = mpScene->getCamera();
    OutputDebugString(L"\n\n= Awaiting camData sending over network... =========\n\n");
    pNetworkManager->SendCameraData(cam, pNetworkManager->mConnectSocket);
    OutputDebugString(L"\n\n= camData sent over network =========\n\n");

    // Send the position texture to server
    int posTexLen = int(NetworkPass::posData.size());
    assert(posTexLen == NetworkPass::posTexWidth * NetworkPass::posTexHeight * 8);
    OutputDebugString(L"\n\n= Awaiting posTex sending over network... =========\n\n");
    pNetworkManager->SendTexture(posTexLen, (char*)&NetworkPass::posData[0], pNetworkManager->mConnectSocket);
    OutputDebugString(L"\n\n= posTex sent over network =========\n\n");

    // Await server to send back the visibility pass texture
    int visTexLen = NetworkPass::posTexWidth * NetworkPass::posTexHeight * 4;
    OutputDebugString(L"\n\n= Awaiting visTex receiving over network... =========\n\n");
    pNetworkManager->RecvTexture(visTexLen, (char*)&NetworkPass::visibilityData[0], pNetworkManager->mConnectSocket);
    OutputDebugString(L"\n\n= visTex received over network =========\n\n");
}

bool NetworkPass::firstServerRender(RenderContext* pRenderContext)
{
    // TODO: By right, we should receive scene as well

    mFirstRender = false;

    auto serverListen = [&]() {
        ResourceManager::mNetworkManager->ListenServer(pRenderContext, mpResManager, mTexWidth, mTexHeight);
    };
    Threading::dispatchTask(serverListen);
    OutputDebugString(L"\n\n= ServerRecv - Network thread dispatched =========\n\n");
    return true;
}

void NetworkPass::executeServerRecv(RenderContext* pRenderContext)
{
    std::unique_lock<std::mutex> lck(NetworkManager::mMutex);

    // Perform the first render steps (start the network thread)
    mFirstRender && firstServerRender(pRenderContext);

    // Wait for the network thread to receive the position texture
    OutputDebugString(L"\n\n= ServerRecv - Awaiting PosTex from client... =========\n\n");
    while (!NetworkManager::mPosTexReceived) 
        NetworkManager::mCvPosTexReceived.wait(lck);
    // Reset to false so that we will need to wait for the network pass to flag it as received
    // before we can continue rendering the next frame
    NetworkManager::mPosTexReceived = false;
    OutputDebugString(L"\n\n= ServerRecv - PosTex received from client =========\n\n");

    // COMMENT
    // Load camera data to scene
    Camera::SharedPtr cam = mpScene->getCamera();
    cam->setPosition(camData[0]);
    cam->setUpVector(camData[1]);
    cam->setTarget(camData[2]);

    std::string camPosStr = std::to_string(camData[0].x) + std::string(", ") + std::to_string(camData[0].y) + std::string(", ") + std::to_string(camData[0].z);
    std::string camUpStr = std::to_string(camData[1].x) + std::string(", ") + std::to_string(camData[1].y) + std::string(", ") + std::to_string(camData[1].z);
    std::string camTargetStr = std::to_string(camData[2].x) + std::string(", ") + std::to_string(camData[2].y) + std::string(", ") + std::to_string(camData[2].z);
    std::string camMsg = std::string("Cam Pos: ") + camPosStr + std::string(", ") + std::string("Cam Up: ") + camUpStr + std::string(", ") + std::string("Cam Target: ") + camTargetStr;
    std::string camFinalMsg = std::string("\n================================ Camera Info:  ") + camMsg + std::string(" ================================\n");
    OutputDebugString(string_2_wstring(camFinalMsg).c_str());

    // Load position texture from CPU to GPU
    Texture::SharedPtr posTex2 = mpResManager->getTexture("WorldPosition2");
    posTex2->apiInitPub(NetworkPass::posData.data(), true);
    OutputDebugString(L"\n\n= ServerRecv - Texture loaded to GPU =========\n\n");

    // After this, the server visibilty pass will render
}

void NetworkPass::executeServerSend(RenderContext* pRenderContext)
{
    OutputDebugString(L"\n\n= ServerSend - VisTex finished rendering =========\n\n");
    // Load visibility texture from GPU to CPU
    Texture::SharedPtr visTex = mpResManager->getTexture("VisibilityBitmap");
    visibilityData = visTex->getTextureData(pRenderContext, 0, 0, "");
    OutputDebugString(L"\n\n= ServerSend - VisTex loaded to CPU =========\n\n");

    // Let the network thread send the visibilty texture
    NetworkManager::mVisTexComplete = true;
    NetworkManager::mCvVisTexComplete.notify_all();
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

