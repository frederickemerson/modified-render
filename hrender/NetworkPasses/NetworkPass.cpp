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
    mpResManager->requestTextureResource("WorldPosition", ResourceFormat::RGBA32Float, ResourceManager::kDefaultFlags, mTexWidth, mTexHeight); // Only for client
    // For server buffers, we are creating them here, so we specify their width/height accordingly
    mpResManager->requestTextureResource("WorldPosition2", ResourceFormat::RGBA32Float, ResourceManager::kDefaultFlags, mTexWidth, mTexHeight);
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
    if (mMode == Mode::ServerUdpSend)
        executeServerUdpSend(pRenderContext);
    else if (mMode == Mode::ServerUdp)
        executeServerUdpRecv(pRenderContext);
    else if (mMode == Mode::ClientUdpSend)
        executeClientUdpSend(pRenderContext);
    // testing new ordering of commands
    else if (mMode == Mode::ClientUdpSendFirst)
        // firstClientSendUdp();
        OutputDebugStringA("Unused mode: ClientUdpSendFirst");
    else // (mMode == Mode::ClientUdp)
        executeClientUdpRecv(pRenderContext);
}

/// <summary>
/// Fetch the texture data from the texture pointer under the given render context.
/// </summary>
/// <param name="pRenderContext">- render context</param>
/// <param name="tex">- texture pointer</param>
/// <returns></returns>
std::vector<uint8_t> NetworkPass::texData(RenderContext* pRenderContext, Texture::SharedPtr tex)
{
    return tex->getTextureData(pRenderContext, 0, 0);
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

void NetworkPass::executeClientUdpSend(RenderContext* pRenderContext)
{
    NetworkManager::SharedPtr pNetworkManager = mpResManager->mNetworkManager;

    // Slight branch optimization over:
    mFirstRender && firstClientRenderUdp(pRenderContext);

    // Signal sending thread to send the camera data
    pNetworkManager->mSpClientCamPosReadyToSend.signal();
}

void NetworkPass::executeServerUdpSend(RenderContext* pRenderContext)
{
    NetworkManager::mSpServerVisTexComplete.signal();
    if (firstServerSend)
    {
        // Start the server sending thread
        auto serverSend = [&]()
        {
            ResourceManager::mNetworkManager->
                SendWhenReadyServerUdp(pRenderContext, mpResManager, mTexWidth, mTexHeight);
        };
        Threading::dispatchTask(serverSend);
        firstServerSend = false;
    }
}

void NetworkPass::executeClientUdpRecv(RenderContext* pRenderContext)
{
    if (firstClientReceive)
    {
        NetworkManager::SharedPtr mpNetworkManager = mpResManager->mNetworkManager;
        // First client listen to be run in sequence
        mpNetworkManager->ListenClientUdp(true, false);

        // Start the client receiving thread
        auto serverSend = [mpNetworkManager]()
        {
            mpNetworkManager->ListenClientUdp(true, true);
        };
        Threading::dispatchTask(serverSend);
        firstClientReceive = false;
    }
}

void NetworkPass::executeServerUdpRecv(RenderContext* pRenderContext)
{
    // Perform the first render steps (start the network thread)
    mFirstRender && firstServerRenderUdp(pRenderContext);

    // Wait if the network thread has not received yt
    NetworkManager::mSpServerCamPosUpdated.wait();

    // Load camera data to scene
    Camera::SharedPtr cam = mpScene->getCamera();
    {
        std::lock_guard lock(NetworkManager::mMutexServerCamData); 
        cam->setPosition(NetworkPass::camData[0]);
        cam->setUpVector(NetworkPass::camData[1]);
        cam->setTarget(NetworkPass::camData[2]);
    }

    // Update the scene. Ideally, this should be baked into RenderingPipeline.cpp and executed
    // before the pass even starts and after the network thread receives the data.
    mpScene->update(pRenderContext, gpFramework->getGlobalClock().getTime());

    // Recalculate, if we could do calculateCameraParameters() instead, we would.
    cam->getViewMatrix();

    OutputDebugString(L"\n\n= ServerUdpRecv - CamPos received from client =========");

    // After this, the server JitteredGBuffer pass will render
}

bool NetworkPass::firstClientRenderUdp(RenderContext* pRenderContext)
{
    NetworkManager::SharedPtr pNetworkManager = mpResManager->mNetworkManager;

    // Send the texture size to the server
    OutputDebugString(L"\n\n= firstClientRenderUdp: Sending width/height over network... =========");
    int32_t widthAndHeight[2];
    widthAndHeight[0] = mpResManager->getWidth();
    widthAndHeight[1] = mpResManager->getHeight();
    
    // Sequence number of 0, size of 8 bytes
    UdpCustomPacketHeader header(0, 8);
    char* dataPtr = reinterpret_cast<char*>(&widthAndHeight);
    pNetworkManager->SendUdpCustom(header, dataPtr, pNetworkManager->mClientUdpSock);
    
    // Next sequence number should be 1
    pNetworkManager->clientSeqNum = 1;
    OutputDebugString(L"\n\n= firstClientRenderUdp: width/height sent over network =========");

    // Populate posTexWidth and Height
    NetworkPass::posTexWidth = mpResManager->getWidth();
    NetworkPass::posTexHeight = mpResManager->getHeight();

    // Start the client sending thread
    auto clientSendWhenReady = [&]()
    {
        ResourceManager::mNetworkManager->SendWhenReadyClientUdp(mpScene);
    };
    Threading::dispatchTask(clientSendWhenReady);

    mFirstRender = false;
    return true;
}

bool NetworkPass::firstServerRenderUdp(RenderContext* pRenderContext)
{
    // TODO: By right, we should receive scene as well

    mFirstRender = false;

    // Wait for first cam data to be received (run in sequence)
    ResourceManager::mNetworkManager->ListenServerUdp(false, true);
    auto serverListen = [&]() {
        // First packet in parallel will take some time to arrive as well
        int numOfTimes = 1;
        for (int i = 0; i < numOfTimes; i++)
        {
            ResourceManager::mNetworkManager->ListenServerUdp(false, true);
        }
        // Afterwards, loop infinitely
        ResourceManager::mNetworkManager->ListenServerUdp(true, false);
    };
    Threading::dispatchTask(serverListen);
    OutputDebugString(L"\n\n= firstServerRenderUdp - Network thread dispatched =========");
    return true;
}
