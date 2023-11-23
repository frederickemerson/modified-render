#include "NetworkClientSendPass.h"


void NetworkClientSendPass::execute(RenderContext* pRenderContext)
{
    ClientNetworkManager::SharedPtr pNetworkManager = mpResManager->mClientNetworkManager;

    // Slight branch optimization over:
    mFirstRender && firstClientRenderUdp(pRenderContext);

    // Signal sending thread to send the camera data
    pNetworkManager->setArtificialLag(mArtificialDelay);
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

void NetworkClientSendPass::renderGui(Gui::Window* pPassWindow)
{
    // Window is marked dirty if any of the configuration is changed.
    int dirty = 0;

    dirty |= (int)pPassWindow->var("Artificial Delay", mArtificialDelay, 0, 20000, 1);

    // If any of our UI parameters changed, let the pipeline know we're doing something different next frame
    if (dirty) setRefreshFlag();
}
