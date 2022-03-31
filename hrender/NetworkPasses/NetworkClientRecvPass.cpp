#include "NetworkClientRecvPass.h"
#include <cmath>

char* NetworkClientRecvPass::clientReadBuffer = new char[VIS_TEX_LEN];
char* NetworkClientRecvPass::clientWriteBuffer = new char[VIS_TEX_LEN];
char* NetworkClientRecvPass::intermediateBuffer = new char[VIS_TEX_LEN];

void NetworkClientRecvPass::execute(RenderContext* pRenderContext)
{
    if (firstClientReceive)
    {
        ClientNetworkManager::SharedPtr pNetworkManager = mpResManager->mClientNetworkManager;

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
        //pNetworkManager->mSpClientSeqTexRecv.wait();
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
    pPassWindow->text("Time for one frame: " + std::to_string(mpResManager->
        mClientNetworkManager->getTimeForOneSequentialFrame()));

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


float cameraWeightUX;
float cameraWeightUY;
float cameraWeightUZ;
float cameraWeightVX;
float cameraWeightVY;
float cameraWeightVZ;
float cameraWeightWX;
float cameraWeightWY;
float cameraWeightWZ;

float networkWeight;
float constant;

void initialiseWeights() {
    // we want initial weights to be such that weight * value = around the same value
    float cameraWeightUX = 1;
    float cameraWeightUY = 1;
    float cameraWeightUZ = 1;
    float cameraWeightVX = 1;
    float cameraWeightVY = 1;
    float cameraWeightVZ = 1;
    float cameraWeightWX = 1;
    float cameraWeightWY = 1;
    float cameraWeightWZ = 1;

    float networkWeight = 1;
    float constant = 1;
}

int predictLoss() {
    // camera data weights
    
    return 0;
}



void fit() {
    int predictedLoss = predictLoss();
    //int actualLoss = calculateLoss();
    


}