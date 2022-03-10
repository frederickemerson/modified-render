#include "NetworkClientRecvPass.h"
#include <cmath>

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
        remainInSequential--;
        if (remainInSequential <= 0) { checkMotionVector(); }
        pNetworkManager->mSpClientNewTexRecv.wait();
    }
    else {
        // decide if we are switching into sequential
        checkMotionVector();
        checkNetworkPing();
    }
}

void NetworkClientRecvPass::renderGui(Gui::Window* pPassWindow)
{
    // Window is marked dirty if any of the configuration is changed.
    //int dirty = 0;

    // Print the name of the buffer we're accumulating from and into.  Add a blank line below that for clarity
    if (sequential) {
        pPassWindow->text("Sequential: Yes");
    }
    else {
        pPassWindow->text("Sequential: No");
    }

    //pPassWindow->text((std::string("total camera movement: ") + std::to_string(dif)).c_str());

    // If any of our UI parameters changed, let the pipeline know we're doing something different next frame
    //if (dirty) setRefreshFlag();
}

inline void NetworkClientRecvPass::checkMotionVector() {
    Camera::SharedPtr cam = mpScene->getCamera();
    const CameraData& cameraData = cam->getData();

    float dif = 0;
    // store cameraU, V, W specifically for GBuffer rendering later
    dif += std::abs(cameraData.cameraU.x - cameraUX);
    dif += std::abs(cameraData.cameraU.y - cameraUY);
    dif += std::abs(cameraData.cameraU.z - cameraUZ);
    dif += std::abs(cameraData.cameraV.x - cameraVX);
    dif += std::abs(cameraData.cameraV.y - cameraVY);
    dif += std::abs(cameraData.cameraV.z - cameraVZ);
    dif += std::abs(cameraData.cameraW.x - cameraWX);
    dif += std::abs(cameraData.cameraW.y - cameraWY);
    dif += std::abs(cameraData.cameraW.z - cameraWZ);

    cameraUX = cameraData.cameraU.x;
    cameraUY = cameraData.cameraU.y;
    cameraUZ = cameraData.cameraU.z;
    cameraVX = cameraData.cameraV.x;
    cameraVY = cameraData.cameraV.y;
    cameraVZ = cameraData.cameraV.z;
    cameraWX = cameraData.cameraW.x;
    cameraWY = cameraData.cameraW.y;
    cameraWZ = cameraData.cameraW.z;

    if (dif > 1000) {
        sequential = true;
        remainInSequential = 20;
    }
    else if (sequential && dif > 300) {
        remainInSequential = 20;
    }
    else if (sequential && dif < 50) {
        sequential = false;
    }
}

inline void NetworkClientRecvPass::checkNetworkPing() {

}