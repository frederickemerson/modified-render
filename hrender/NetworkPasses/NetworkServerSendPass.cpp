#include "NetworkServerSendPass.h"


void NetworkServerSendPass::execute(RenderContext* pRenderContext)
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