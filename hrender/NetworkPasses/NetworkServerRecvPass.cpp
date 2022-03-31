#include "NetworkServerRecvPass.h"

std::vector<std::array<float3, 3>> NetworkServerRecvPass::clientCamData;
std::array<std::mutex, MAX_NUM_CLIENT> NetworkServerRecvPass::mutexForCamData = { std::mutex(),
        std::mutex() , std::mutex() , std::mutex() };
std::queue<int> NetworkServerRecvPass::frameNumRendered;

void NetworkServerRecvPass::execute(RenderContext* pRenderContext)
{
    // Perform the first render steps (start the network thread)
    mFirstRender&& firstServerRenderUdp(pRenderContext);

    // Wait if the network thread has not received yt
    ServerNetworkManager::mSpServerCamPosUpdated.wait();

    // Load camera data to scene
    Camera::SharedPtr cam = mpScene->getCamera();
    int clientIndex = getClientToRender();
    {
        std::lock_guard lock(mutexForCamData[clientIndex]);
        frameNumRendered.push(ResourceManager::mServerNetworkManager->clientFrameNum[clientIndex]);
        auto camData = NetworkServerRecvPass::clientCamData[clientIndex];
        cam->setPosition(camData[0]);
        cam->setUpVector(camData[1]);
        cam->setTarget(camData[2]);
    }
    // add this client to NetworkManager's send client queue
    ResourceManager::mServerNetworkManager->sendClientQueue.push(clientIndex);

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
    mFirstRender = false;
    auto serverListen = [&]() {
        ResourceManager::mServerNetworkManager->ListenServerUdp();
    };
    Threading::dispatchTask(serverListen);
    OutputDebugString(L"\n\n= firstServerRenderUdp - Network thread dispatched =========");
    return true;
}

int NetworkServerRecvPass::getClientToRender()
{
    size_t numOfActiveClients = ResourceManager::mServerNetworkManager->mClientAddresses.size();
    clientIndexToRender = (clientIndexToRender + 1) % numOfActiveClients;
    // TODO, WE SHOULD GO NEXT IF THE CAM POS HASTN ARRIVED
    ServerNetworkManager::mClientCamPosUpdated[clientIndexToRender].wait();
    return clientIndexToRender;
}
