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