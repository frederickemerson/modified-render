#include "NetworkServerSendPass.h"

char* NetworkServerSendPass::intermediateBuffer = new char[VIS_TEX_LEN + REF_TEX_LEN + AO_TEX_LEN];

void NetworkServerSendPass::execute(RenderContext* pRenderContext)
{
    ResourceManager::mServerNetworkManager->setArtificialLag(mArtificialDelay);
    ServerNetworkManager::mSpServerVisTexComplete.signal();
    if (firstServerSend)
    {
        // Start the server sending thread
        auto serverSend = [&]()
        {
            ResourceManager::mServerNetworkManager->
                SendWhenReadyServerUdp(pRenderContext, mpResManager, mTexWidth, mTexHeight);
        };
        Threading::dispatchTask(serverSend);
        firstServerSend = false;
    }
}

void NetworkServerSendPass::renderGui(Gui::Window* pPassWindow)
{
    // Window is marked dirty if any of the configuration is changed.
    int dirty = 0;

    dirty |= (int)pPassWindow->var("Artificial Delay", mArtificialDelay, 0, 20000, 1);

    // If any of our UI parameters changed, let the pipeline know we're doing something different next frame
    if (dirty) setRefreshFlag();
}
