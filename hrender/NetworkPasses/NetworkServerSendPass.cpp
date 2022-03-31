#include "NetworkServerSendPass.h"

char* NetworkServerSendPass::intermediateBuffer = new char[VIS_TEX_LEN];

void NetworkServerSendPass::execute(RenderContext* pRenderContext)
{
    ServerNetworkManager::mSpServerVisTexComplete.signal();
    if (firstServerSend)
    {
        // Start the server sending thread
        auto serverSend = [&]()
        {
            ResourceManager::mServerNetworkManager->
                SendWhenReadyServerUdp(mTexWidth, mTexHeight);
        };
        Threading::dispatchTask(serverSend);
        firstServerSend = false;
    }
}