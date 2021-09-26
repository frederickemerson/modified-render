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
#include "NetworkUtils.h"

//std::vector<uint8_t> NetworkPass::normData = std::vector<uint8_t>();
std::vector<uint8_t> NetworkPass::posData = std::vector<uint8_t>(POS_TEX_LEN, 0);
int NetworkPass::posTexWidth = 0;
int NetworkPass::posTexHeight = 0;

//std::vector<uint8_t> NetworkPass::gBufData = std::vector<uint8_t>();
std::vector<uint8_t> NetworkPass::visibilityData = std::vector<uint8_t>(VIS_TEX_LEN, 0);
std::vector<uint8_t> NetworkPass::compressionBuffer = std::vector<uint8_t>(VIS_TEX_LEN, 0); // for compression
uint8_t* NetworkPass::pVisibilityData = NetworkPass::visibilityData.data(); // for server side GPU-CPU trsf of visibilityBuffer, stores location of data, changes every frame
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
    if (mMode == Mode::ServerSend)
        executeServerSend(pRenderContext);
    else if (mMode == Mode::Server)
        executeServerRecv(pRenderContext);
    else if (mMode == Mode::ClientSend)
        executeClientSend(pRenderContext);
    else if (mMode == Mode::Client)
        executeClientRecv(pRenderContext);
    else if (mMode == Mode::ServerUdpSend)
        // TODO: Change to UDP-specific function
        executeServerSend(pRenderContext);
    else if (mMode == Mode::ServerUdp)
        executeServerUdpRecv(pRenderContext);
    else if (mMode == Mode::ClientUdpSend)
        executeClientUdpSend(pRenderContext);
    // testing new ordering of commands
    else if (mMode == Mode::ClientUdpSendFirst)
        firstClientSendUdp();
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

/// <summary>
/// Executes on the first frame of program run on client side, responsible to send datas that
/// are only required to be sent once. Currently sends the width and height of the
/// window to server, and internally keep track of it.
/// </summary>
/// <param name="pRenderContext">- render context</param>
/// <returns></returns>
bool NetworkPass::firstClientRender(RenderContext* pRenderContext)
{
    NetworkManager::SharedPtr pNetworkManager = mpResManager->mNetworkManager;

    // Send the texture size to the server
    OutputDebugString(L"\n\n= Awaiting width/height sending over network... =========");
    pNetworkManager->SendInt(mpResManager->getWidth(), pNetworkManager->mConnectSocket);
    pNetworkManager->SendInt(mpResManager->getHeight(), pNetworkManager->mConnectSocket);
    OutputDebugString(L"\n\n= width/height sent over network =========");

    // TODO: Send scene

    // Populate posTexWidth and Height
    NetworkPass::posTexWidth = mpResManager->getWidth();
    NetworkPass::posTexHeight = mpResManager->getHeight();

    mFirstRender = false;

    return true;
}

/// <summary>
/// Executes on every frame of program run on client side, responsible to send datas that
/// are required to be sent on every frame. Currently sends the camera data to the server.
/// </summary>
/// <param name="pRenderContext">- render context</param>
void NetworkPass::executeClientSend(RenderContext* pRenderContext)
{
    NetworkManager::SharedPtr pNetworkManager = mpResManager->mNetworkManager;

    // Slight branch optimization over:
    mFirstRender&& firstClientRender(pRenderContext);

    // Send camera data from client to server
    Camera::SharedPtr cam = mpScene->getCamera();
    OutputDebugString(L"\n\n= Awaiting camData sending over network... =========");
    pNetworkManager->SendCameraData(cam, pNetworkManager->mConnectSocket);
    OutputDebugString(L"\n\n= camData sent over network =========");
}

/// <summary>
/// Executes on every frame of program run on client side, responsible to receive datas that
/// are required to be received on every frame. Currently receives the visibility bitmap
/// from the server.
/// </summary>
/// <param name="pRenderContext">- render context</param>
void NetworkPass::executeClientRecv(RenderContext* pRenderContext)
{
    NetworkManager::SharedPtr pNetworkManager = mpResManager->mNetworkManager;

    // Await server to send back the visibility pass texture
    int visTexLen = NetworkPass::posTexWidth * NetworkPass::posTexHeight * 4;
    OutputDebugString(L"\n\n= Awaiting visTex receiving over network... =========");
    pNetworkManager->RecvTexture(visTexLen, (char*)&NetworkPass::visibilityData[0], pNetworkManager->mConnectSocket);
    OutputDebugString(L"\n\n= visTex received over network =========");
}

/// <summary>
/// Executes on the first frame of program run on server side, responsible to receive datas that
/// are only required to be received once. Currently is set to start a separate thread that will accept
/// client connection.
/// </summary>
/// <param name="pRenderContext">- render context</param>
/// <returns></returns>
bool NetworkPass::firstServerRender(RenderContext* pRenderContext)
{
    // TODO: By right, we should receive scene as well

    mFirstRender = false;

    auto serverListen = [&]() {
        ResourceManager::mNetworkManager->ListenServer(pRenderContext, mpResManager, mTexWidth, mTexHeight);
    };
    Threading::dispatchTask(serverListen);
    OutputDebugString(L"\n\n= ServerRecv - Network thread dispatched =========");
    return true;
}

/// <summary>
/// Executes on every frame of program run on server side, responsible to receive datas that
/// are required to be received on every frame. Currently receives the camera data from the client.
/// </summary>
/// <param name="pRenderContext">- render context</param>
void NetworkPass::executeServerRecv(RenderContext* pRenderContext)
{
    std::unique_lock<std::mutex> lck(NetworkManager::mMutex);

    // Perform the first render steps (start the network thread)
    mFirstRender&& firstServerRender(pRenderContext);

    // Wait for the network thread to receive the cameraPosition
    OutputDebugString(L"\n\n= ServerRecv - Awaiting camPos from client... =========");
    while (!NetworkManager::mCamPosReceived)
        NetworkManager::mCvCamPosReceived.wait(lck);

    // Load camera data to scene
    Camera::SharedPtr cam = mpScene->getCamera();
    cam->setPosition(NetworkPass::camData[0]);
    cam->setUpVector(NetworkPass::camData[1]);
    cam->setTarget(NetworkPass::camData[2]);

    // Update the scene. Ideally, this should be baked into RenderingPipeline.cpp and executed
    // before the pass even starts and after the network thread receives the data.
    mpScene->update(pRenderContext, gpFramework->getGlobalClock().getTime());

    // Reset to false so that we will need to wait for the network pass to flag it as received
    // before we can continue rendering the next frame
    NetworkManager::mCamPosReceived = false;

    // Recalculate, if we could do calculateCameraParameters() instead, we would.
    cam->getViewMatrix();

    OutputDebugString(L"\n\n= ServerRecv - CamPos received from client =========");

    // After this, the server JitteredGBuffer pass will render
}

/// <summary>
/// Executes on every frame of program run on server side, responsible to send datas that
/// are required to be sent on every frame. Server send only notifies the mutex such that
/// NetworkManager can send the actual data.
/// </summary>
/// <param name="pRenderContext">- render context</param>
void NetworkPass::executeServerSend(RenderContext* pRenderContext)
{
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

void NetworkPass::executeClientUdpSend(RenderContext* pRenderContext)
{
    NetworkManager::SharedPtr pNetworkManager = mpResManager->mNetworkManager;

    // Slight branch optimization over:
    mFirstRender && firstClientRenderUdp(pRenderContext);

    // Send camera data from client to server
    Camera::SharedPtr cam = mpScene->getCamera();

    // store cameraU, V, W specifically for GBuffer rendering later
    const CameraData& cameraData = cam->getData();
    pNetworkManager->cameraU = cameraData.cameraU;
    pNetworkManager->cameraV = cameraData.cameraV;
    pNetworkManager->cameraW = cameraData.cameraW;

    OutputDebugString(L"\n\n= Awaiting camData sending over network... =========");
    pNetworkManager->SendCameraDataUdp(cam, pNetworkManager->mClientUdpSock);
    OutputDebugString(L"\n\n= camData sent over network =========");
}

void NetworkPass::firstClientSendUdp() {
    if (!mFirstRender) {
        return;
    }
    
    NetworkManager::SharedPtr pNetworkManager = mpResManager->mNetworkManager;

    // Send the texture size to the server
    OutputDebugString(L"\n\n= firstClientRenderUdp: Sending width/height over network... =========");
    int32_t widthAndHeight[2];
    widthAndHeight[0] = mpResManager->getWidth();
    widthAndHeight[1] = mpResManager->getHeight();
    // Sequence number of 0
    UdpCustomPacket packet(0, 8, reinterpret_cast<uint8_t*>(&widthAndHeight));
    pNetworkManager->SendUdpCustom(packet, pNetworkManager->mClientUdpSock);
    packet.releaseDataPointer();
    // Next sequence number should be 1
    pNetworkManager->currentSeqNum = 1;
    OutputDebugString(L"\n\n= firstClientRenderUdp: width/height sent over network =========");

    // Initialise the latest texture cache
    int visTexSize = widthAndHeight[0] * widthAndHeight[1] * 4;
    pNetworkManager->latestTextureData = new char[visTexSize];

    // Populate posTexWidth and Height
    NetworkPass::posTexWidth = mpResManager->getWidth();
    NetworkPass::posTexHeight = mpResManager->getHeight();
    mFirstRender = false;

    // Send camera data from client to server
    Camera::SharedPtr cam = mpScene->getCamera();
    OutputDebugString(L"\n\n= Awaiting camData sending over network... =========");
    pNetworkManager->SendCameraDataUdp(cam, pNetworkManager->mClientUdpSock);
    OutputDebugString(L"\n\n= camData sent over network =========");
}

void NetworkPass::executeClientUdpRecv(RenderContext* pRenderContext)
{
    // dont render the first time because visibilityBuffer doesnt exist yet (new ordering sends camera Data after receive visibilityBuffer)
    /*if (mFirstRender) {
        mFirstRender = false;
        return;
    }*/
    NetworkManager::SharedPtr pNetworkManager = mpResManager->mNetworkManager;

    // Await server to send back the visibility pass texture
    int visTexLen = NetworkPass::posTexWidth * NetworkPass::posTexHeight * 4;
    OutputDebugString(L"\n\n= Awaiting visTex receiving over network... =========");
    FrameData rcvdFrameData = { visTexLen, 0, 0 };
    
    char* toRecvData;
    if (NetworkManager::mCompression) {
        toRecvData = (char*)&NetworkPass::compressionBuffer[0];
    }
    else {
        toRecvData = (char*)&NetworkPass::visibilityData[0];
    }


    if (firstClientReceive)
    {        
        // Need to take a while to wait for the server,
        // so we use a longer time out for the first time
        pNetworkManager->RecvTextureUdp(rcvdFrameData,
                                        toRecvData,
                                        pNetworkManager->mClientUdpSock,
                                        UDP_FIRST_TIMEOUT_MS);
        // Store the time when the first frame was received
        // (server sends timestamps relative to the time when the first frame was fully rendered)
        startTime = getCurrentTime();
        firstClientReceive = false;
    }
    else
    {
        pNetworkManager->RecvTextureUdp(rcvdFrameData,
                                        toRecvData,
                                        pNetworkManager->mClientUdpSock);
        
        // TODO: make use of timestamp to delay if needed
        std::chrono::milliseconds currentTime = getComparisonTimestamp();
        std::chrono::milliseconds timeDifference = currentTime - std::chrono::milliseconds(rcvdFrameData.timestamp);
        if (timeDifference > std::chrono::milliseconds::zero())
        {
            char sleepMessage[89];
            sprintf(sleepMessage, "\nSleeping for %d=========", static_cast<int>(timeDifference.count()));
            OutputDebugStringA(sleepMessage);
            std::this_thread::sleep_for(timeDifference);
        }
        else
        {
            char sleepMessage[89];
            sprintf(sleepMessage, "\nBehind by %d=========", static_cast<int>(timeDifference.count()));
            OutputDebugStringA(sleepMessage);
        }
    }
    OutputDebugString(L"\n\n= visTex received over network =========");
    char frameDataMessage[89];
    sprintf(frameDataMessage, "\nFrameData: Number: %d, Size: %d, Time: %d\n",
            rcvdFrameData.frameNumber, rcvdFrameData.frameSize, rcvdFrameData.timestamp);
    
    // if compress
    if (NetworkManager::mCompression) {
        pNetworkManager->DecompressTextureLZ4(visTexLen, (char*)&NetworkPass::visibilityData[0], rcvdFrameData.frameSize, toRecvData);
    }
    
    OutputDebugStringA(frameDataMessage);
}

void NetworkPass::executeServerUdpRecv(RenderContext* pRenderContext)
{
    std::unique_lock<std::mutex> lck(NetworkManager::mMutex);

    // Perform the first render steps (start the network thread)
    mFirstRender && firstServerRenderUdp(pRenderContext);

    // Wait for the network thread to receive the cameraPosition
    OutputDebugString(L"\n\n= ServerRecv - Awaiting camPos from client... =========");
    while (!NetworkManager::mCamPosReceived)
        NetworkManager::mCvCamPosReceived.wait(lck);

    // Load camera data to scene
    Camera::SharedPtr cam = mpScene->getCamera();
    cam->setPosition(NetworkPass::camData[0]);
    cam->setUpVector(NetworkPass::camData[1]);
    cam->setTarget(NetworkPass::camData[2]);

    // Update the scene. Ideally, this should be baked into RenderingPipeline.cpp and executed
    // before the pass even starts and after the network thread receives the data.
    mpScene->update(pRenderContext, gpFramework->getGlobalClock().getTime());

    // Reset to false so that we will need to wait for the network pass to flag it as received
    // before we can continue rendering the next frame
    NetworkManager::mCamPosReceived = false;

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
    // Sequence number of 0
    UdpCustomPacket packet(0, 8, reinterpret_cast<uint8_t*>(&widthAndHeight));
    pNetworkManager->SendUdpCustom(packet, pNetworkManager->mClientUdpSock);
    packet.releaseDataPointer();
    // Next sequence number should be 1
    pNetworkManager->currentSeqNum = 1;
    OutputDebugString(L"\n\n= firstClientRenderUdp: width/height sent over network =========");

    // Initialise the latest texture cache
    int visTexSize = widthAndHeight[0] * widthAndHeight[1] * 4;
    pNetworkManager->latestTextureData = new char[visTexSize];

    // Populate posTexWidth and Height
    NetworkPass::posTexWidth = mpResManager->getWidth();
    NetworkPass::posTexHeight = mpResManager->getHeight();
    mFirstRender = false;
    return true;
}

bool NetworkPass::firstServerRenderUdp(RenderContext* pRenderContext)
{
    // TODO: By right, we should receive scene as well

    mFirstRender = false;

    auto serverListen = [&]() {
        ResourceManager::mNetworkManager->ListenServerUdp(pRenderContext, mpResManager, mTexWidth, mTexHeight);
    };
    Threading::dispatchTask(serverListen);
    OutputDebugString(L"\n\n= ServerRecv - Network thread dispatched =========");
    return true;
}

std::chrono::milliseconds NetworkPass::getComparisonTimestamp()
{
    return getCurrentTime() - startTime;
}
