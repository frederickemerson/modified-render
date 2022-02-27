#include "NetworkClientRecvWaitingPass.h"


void NetworkClientRecvWaitingPass::execute(RenderContext* pRenderContext)
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
    else {
        NetworkManager::SharedPtr mpNetworkManager = mpResManager->mNetworkManager;
        mpNetworkManager->mSpClientNewTexRecv.wait();
    }
}