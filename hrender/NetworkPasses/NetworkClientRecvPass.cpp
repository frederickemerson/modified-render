#include "NetworkClientRecvPass.h"


void NetworkClientRecvPass::execute(RenderContext* pRenderContext)
{
    if (firstClientReceive)
    {
        NetworkManager::SharedPtr pNetworkManager = mpResManager->mNetworkManager;

        // First client listen to be run in sequence
        pNetworkManager->ListenClientUdp(true, false);

        // Start the client receiving thread
        auto serverSend = [pNetworkManager]()
        {
            pNetworkManager->ListenClientUdp(true, true);
        };
        Threading::dispatchTask(serverSend);
        firstClientReceive = false;
    }
    else if (sequential) {
        NetworkManager::SharedPtr pNetworkManager = mpResManager->mNetworkManager;
        pNetworkManager->mSpClientNewTexRecv.wait();
    }
    // decide if we are switching to or out of sequential
    checkMotionVector();
    checkNetworkPing();
}

void NetworkClientRecvPass::renderGui(Gui::Window* pPassWindow)
{
    // Window is marked dirty if any of the configuration is changed.
    int dirty = 0;

    // Print the name of the buffer we're accumulating from and into.  Add a blank line below that for clarity
    pPassWindow->text("Sequential:   " + sequential ? "Yes" : "No");

    auto cam = mpScene->getCamera();
    const CameraData& cameraData = cam->getData();

    // store cameraU, V, W specifically for GBuffer rendering later
    cameraUX = cameraData.cameraU.x;
    cameraUY = cameraData.cameraU.y;
    cameraUZ = cameraData.cameraU.z;
    cameraVX = cameraData.cameraV.x;
    cameraVY = cameraData.cameraV.y;
    cameraVZ = cameraData.cameraV.z;
    cameraWX = cameraData.cameraW.x;
    cameraWY = cameraData.cameraW.y;
    cameraWZ = cameraData.cameraW.z;

    pPassWindow->text((std::string("U.x: ") + std::to_string(cameraData.cameraU.x)).c_str());
    pPassWindow->text((std::string("U.y: ") + std::to_string(cameraData.cameraU.y)).c_str());
    pPassWindow->text((std::string("U.z: ") + std::to_string(cameraData.cameraU.z)).c_str());
    pPassWindow->text((std::string("V.x: ") + std::to_string(cameraData.cameraV.x)).c_str());
    pPassWindow->text((std::string("V.y: ") + std::to_string(cameraData.cameraV.y)).c_str());
    pPassWindow->text((std::string("V.z: ") + std::to_string(cameraData.cameraV.z)).c_str());    
    pPassWindow->text((std::string("W.x: ") + std::to_string(cameraData.cameraW.x)).c_str());
    pPassWindow->text((std::string("W.y: ") + std::to_string(cameraData.cameraW.y)).c_str());
    pPassWindow->text((std::string("W.z: ") + std::to_string(cameraData.cameraW.z)).c_str());

    // If any of our UI parameters changed, let the pipeline know we're doing something different next frame
    if (dirty) setRefreshFlag();
}

inline void NetworkClientRecvPass::checkMotionVector() {
    
}

inline void NetworkClientRecvPass::checkNetworkPing() {

}