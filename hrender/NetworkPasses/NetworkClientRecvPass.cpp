#include "NetworkClientRecvPass.h"
#include <cmath>

char* NetworkClientRecvPass::clientReadBuffer = new char[VIS_TEX_LEN];
char* NetworkClientRecvPass::clientWriteBuffer = new char[VIS_TEX_LEN];

void NetworkClientRecvPass::execute(RenderContext* pRenderContext)
{
    if (firstClientReceive)
    {
        ClientNetworkManager::SharedPtr pNetworkManager = mpResManager->mClientNetworkManager;

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
    if (!bSwitching) {
        return;
    }

    if (sequential) {
        ClientNetworkManager::SharedPtr pNetworkManager = mpResManager->mClientNetworkManager;
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
    int dirty = 0;

    // Determine whether we are going prediction
    dirty |= (int)pPassWindow->checkbox(bSwitching ? "Automatic Switching" : "Not Switching", bSwitching);

    // Print the name of the buffer we're accumulating from and into.  Add a blank line below that for clarity
    if (sequential) {
        pPassWindow->text("Sequential: Yes");
    }
    else {
        pPassWindow->text("Sequential: No");
    }
    pPassWindow->text("Total camera change: " + std::to_string(totalCameraChange));
    dirty |= (int)pPassWindow->var("Low Threshold", lowThreshold, 0, 20000, 1);
    dirty |= (int)pPassWindow->var("Mid Threshold", midThreshold, 0, 20000, 1);
    dirty |= (int)pPassWindow->var("High Threshold", highThreshold, 0, 20000, 1);

   
    // If any of our UI parameters changed, let the pipeline know we're doing something different next frame
    if (dirty) setRefreshFlag();
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

    totalCameraChange = dif;

    cameraUX = cameraData.cameraU.x;
    cameraUY = cameraData.cameraU.y;
    cameraUZ = cameraData.cameraU.z;
    cameraVX = cameraData.cameraV.x;
    cameraVY = cameraData.cameraV.y;
    cameraVZ = cameraData.cameraV.z;
    cameraWX = cameraData.cameraW.x;
    cameraWY = cameraData.cameraW.y;
    cameraWZ = cameraData.cameraW.z;

    if (dif > highThreshold) {
        sequential = true;
        remainInSequential = 20;
    }
    else if (sequential && dif > midThreshold) {
        remainInSequential = 20;
    }
    else if (sequential && dif < lowThreshold) {
        sequential = false;
    }
}

inline void NetworkClientRecvPass::checkNetworkPing() {

}