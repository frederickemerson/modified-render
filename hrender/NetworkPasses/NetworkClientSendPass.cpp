#include "NetworkClientSendPass.h"


void NetworkClientSendPass::execute(RenderContext* pRenderContext)
{
    ClientNetworkManager::SharedPtr pNetworkManager = mpResManager->mClientNetworkManager;

    // Slight branch optimization over:
    mFirstRender && firstClientRenderUdp(pRenderContext);

    // Signal sending thread to send the camera data
    pNetworkManager->mSpClientCamPosReadyToSend.signal();
}

bool NetworkClientSendPass::firstClientRenderUdp(RenderContext* pRenderContext)
{
    ClientNetworkManager::SharedPtr pNetworkManager = mpResManager->mClientNetworkManager;

    //// Send the texture size to the server
    //OutputDebugString(L"\n\n= firstClientRenderUdp: Sending width/height over network... =========");
    //int32_t widthAndHeight[2];
    //widthAndHeight[0] = mpResManager->getWidth();
    //widthAndHeight[1] = mpResManager->getHeight();

    //// Sequence number of 0, size of 8 bytes
    //UdpCustomPacketHeader header(0, 8);
    //char* dataPtr = reinterpret_cast<char*>(&widthAndHeight);
    //pNetworkManager->SendUdpCustom(header, dataPtr, pNetworkManager->mClientUdpSock);

    // Next sequence number should be 1
    pNetworkManager->clientSeqNum = 1;
    OutputDebugString(L"\n\n= firstClientRenderUdp: width/height sent over network =========");

    // Start the client sending thread
    auto clientSendWhenReady = [&]()
    {
        ResourceManager::mClientNetworkManager->SendWhenReadyClientUdp(mpScene);
    };
    Threading::dispatchTask(clientSendWhenReady);

    mFirstRender = false;
    return true;
}