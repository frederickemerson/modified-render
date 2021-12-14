#include "NetworkServerRecvPass.h"


void NetworkServerRecvPass::execute(RenderContext* pRenderContext)
{
    // Perform the first render steps (start the network thread)
    mFirstRender&& firstServerRenderUdp(pRenderContext);

    // Wait if the network thread has not received yt
    NetworkManager::mSpServerCamPosUpdated.wait();

    // Load camera data to scene
    Camera::SharedPtr cam = mpScene->getCamera();
    {
        std::lock_guard lock(NetworkManager::mMutexServerCamData);
        cam->setPosition(NetworkPass::camData[0]);
        cam->setUpVector(NetworkPass::camData[1]);
        cam->setTarget(NetworkPass::camData[2]);
    }

    // Update the scene. Ideally, this should be baked into RenderingPipeline.cpp and executed
    // before the pass even starts and after the network thread receives the data.
    mpScene->update(pRenderContext, gpFramework->getGlobalClock().getTime());

    // Recalculate, if we could do calculateCameraParameters() instead, we would.
    cam->getViewMatrix();

    OutputDebugString(L"\n\n= ServerUdpRecv - CamPos received from client =========");

    // After this, the server JitteredGBuffer pass will render
}

bool NetworkServerRecvPass::firstServerRenderUdp(RenderContext* pRenderContext)
{
    // TODO: By right, we should receive scene as well

    mFirstRender = false;

    // Wait for first cam data to be received (run in sequence)
    ResourceManager::mNetworkManager->ListenServerUdp(false, true);
    auto serverListen = [&]() {
        // First packet in parallel will take some time to arrive as well
        int numOfTimes = 1;
        for (int i = 0; i < numOfTimes; i++)
        {
            ResourceManager::mNetworkManager->ListenServerUdp(false, true);
        }
        // Afterwards, loop infinitely
        ResourceManager::mNetworkManager->ListenServerUdp(true, false);
    };
    Threading::dispatchTask(serverListen);
    OutputDebugString(L"\n\n= firstServerRenderUdp - Network thread dispatched =========");
    return true;
}