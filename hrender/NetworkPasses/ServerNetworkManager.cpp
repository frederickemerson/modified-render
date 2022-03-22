#include "../NetworkPasses/NetworkServerRecvPass.h"
#include "../NetworkPasses/NetworkClientRecvPass.h"

#include "ServerNetworkManager.h"

// UDP Server
std::array<Semaphore, 4> ServerNetworkManager::mClientCamPosUpdated = {Semaphore(false), 
        Semaphore(false) , Semaphore(false) , Semaphore(false) }; // max num clients = 4
Semaphore ServerNetworkManager::mSpServerVisTexComplete(false);
Semaphore ServerNetworkManager::mSpServerCamPosUpdated(false);
std::mutex ServerNetworkManager::mMutexServerVisTexRead;

bool ServerNetworkManager::SetUpServerUdp(PCSTR port, int& outTexWidth, int& outTexHeight)
{
    WSADATA wsa;

    //Initialise winsock
    OutputDebugString(L"\n\n= Pre-Falcor Init - ServerNetworkManager::SetUpServerUdp - Initialising Winsock... =========");
    if (WSAStartup(MAKEWORD(2, 2), &wsa) != 0)
    {
        char buffer[70];
        sprintf(buffer, "\n\n= Pre-Falcor Init - WSAStartup failed with error: %d", WSAGetLastError());
        OutputDebugStringA(buffer);
        exit(EXIT_FAILURE);
    }
    OutputDebugString(L"\n\n= Pre-Falcor Init - Initialised. =========");

    //Create a socket
    if ((mServerUdpSock = socket(AF_INET, SOCK_DGRAM, IPPROTO_UDP)) == INVALID_SOCKET)
    {
        char buffer[65];
        sprintf(buffer, "\n\n= Pre-Falcor Init - Could not create socket: %d", WSAGetLastError());
        OutputDebugStringA(buffer);
    }
    OutputDebugString(L"\n\n= Pre-Falcor Init - Socket created. =========\n");

    //Prepare the sockaddr_in structure
    mServer.sin_family = AF_INET;
    mServer.sin_addr.s_addr = INADDR_ANY;
    mServer.sin_port = htons((u_short)std::strtoul(port, NULL, 0));
    memset(&(mServer.sin_zero), 0, 8);

    //Bind
    if (bind(mServerUdpSock, (struct sockaddr*) & mServer, sizeof(mServer)) == SOCKET_ERROR)
    {
        char buffer[69];
        sprintf(buffer, "\n\n= Pre-Falcor Init - Bind failed with error code: %d", WSAGetLastError());
        OutputDebugStringA(buffer);
        exit(EXIT_FAILURE);
    }
    OutputDebugString(L"\n\n= Pre-Falcor Init - UDP SOCKET SETUP COMPLETE =========");

    // for now, we only accept and hardcode the width and height
    outTexWidth = 1920;
    outTexHeight = 1080;

    char printWidthHeight[52];
    sprintf(printWidthHeight, "\nWidth: %d\nHeight: %d", outTexWidth, outTexHeight);
    OutputDebugStringA(printWidthHeight);

    return true;
}

bool ServerNetworkManager::ListenServerUdp(bool executeForever, bool useLongTimeout)
{
    // Receive until the peer shuts down the connection
    do
    {
        std::chrono::time_point startOfFrame = std::chrono::system_clock::now();
        // Receive the camera position from the sender
        OutputDebugString(L"\n\n= NetworkThread - Awaiting camData receiving over network... =========");
        // Mutex will be locked in RecvCameraDataUdp
        //const auto delayStartTime = std::chrono::system_clock::now();            // Artificial Delay
        RecvCameraDataUdp(NetworkServerRecvPass::clientCamData,
            NetworkServerRecvPass::mutexForCamData,
                          mServerUdpSock,
                          useLongTimeout);
        OutputDebugString(L"\n\n= NetworkThread - camData received over network =========");

        std::chrono::time_point endOfFrame = std::chrono::system_clock::now();
        std::chrono::duration<double> diff = endOfFrame - startOfFrame;
        char printFps[102];
        sprintf(printFps, "\n\n= ListenServerUdp - Frame took %.10f s, estimated FPS: %.2f =========", diff.count(), getFps(diff));
        OutputDebugStringA(printFps);
    }
    while (executeForever);

    return true;
}

void ServerNetworkManager::SendWhenReadyServerUdp(
    RenderContext* pRenderContext,
    std::shared_ptr<ResourceManager> pResManager,
    int texWidth,
    int texHeight)
{
    int numFramesRendered = 0;
    // Keep track of time
    std::chrono::milliseconds timeOfFirstFrame;

    while (true)
    {
        std::chrono::time_point startOfFrame = std::chrono::system_clock::now();
        std::string frameMsg = std::string("\n\n================================ Frame ") + std::to_string(++numFramesRendered) + std::string(" ================================");
        OutputDebugString(string_2_wstring(frameMsg).c_str());

        // Allow rendering using the camPos to begin, and wait for visTex to complete rendering
        OutputDebugString(L"\n\n= NetworkThread - Awaiting visTex to finish rendering... =========");
        mSpServerVisTexComplete.wait();
        OutputDebugString(L"\n\n= NetworkThread - VisTex finished rendering. Awaiting visTex sending over network... =========");

        {
            std::lock_guard lock(mMutexServerVisTexRead);
            char* toSendData = mGetInputBuffer();

            // The size of the actual Buffer
            // that is given by Falcor is less then VIS_TEX_LEN
            // 
            // The actual size is the screen width and height * 4
            // We send VIS_TEX_LEN but we need to compress with the actual
            // size to prevent reading outside of the Falcor Buffer
            int visTexSizeActual = texWidth * texHeight * 4;
            int toSendSize = mGetInputBufferSize();

            // Send the visBuffer back to the sender
            // Generate timestamp
            std::chrono::milliseconds currentTime = getCurrentTime();
            int timestamp = static_cast<int>((currentTime - timeOfFirstFrame).count());
            SendTextureUdp({ toSendSize, numFramesRendered, timestamp },
                           toSendData,
                           mServerUdpSock);
        }

        OutputDebugString(L"\n\n= NetworkThread - visTex sent over network =========");
        std::string endMsg = std::string("\n\n================================ Frame ") + std::to_string(numFramesRendered) + std::string(" COMPLETE ================================");
        OutputDebugString(string_2_wstring(endMsg).c_str());
        
        if (numFramesRendered == 1)
        {
            timeOfFirstFrame = getCurrentTime();
        }

        std::chrono::time_point endOfFrame = std::chrono::system_clock::now();
        std::chrono::duration<double> diff = endOfFrame - startOfFrame;
        char printFps[109];
        sprintf(printFps, "\n\n= SendWhenReadyServerUdp - Frame took %.10f s, estimated FPS: %.2f =========", diff.count(), getFps(diff));
        OutputDebugStringA(printFps);
    }
}

bool ServerNetworkManager::CloseServerConnectionUdp()
{
    closesocket(mServerUdpSock);
    WSACleanup();
    return true;
}

void ServerNetworkManager::SendTextureUdp(FrameData frameData, char* sendTexData, SOCKET& socketUdp)
{
    int clientIndexToSend = sendClientQueue.front();
    sendClientQueue.pop();

    // Variable splitSize controls the size of the split packets
    int32_t splitSize = UdpCustomPacket::maxPacketSize;
    int16_t numOfFramePackets = frameData.frameSize / splitSize +
                                ((frameData.frameSize % splitSize > 0) ? 1 : 0);
    
    // Split the frame data and send
    int currentOffset = 0;
    for (int32_t amountLeft = frameData.frameSize; amountLeft > 0; amountLeft -= splitSize)
    {
        int32_t size = std::min(amountLeft, UdpCustomPacket::maxPacketSize);                                  
        UdpCustomPacketHeader texHeader(serverSeqNum[clientIndexToSend], size, frameData.frameNumber,
                                        numOfFramePackets, frameData.timestamp);

        if (!SendUdpCustom(texHeader, &sendTexData[currentOffset], clientIndexToSend, socketUdp))
        {
            char buffer[70];
            sprintf(buffer, "\n\n= SendTextureUdp: Failed to send packet %d =========",
                    texHeader.sequenceNumber);
            OutputDebugStringA(buffer);
            return;
        }

        serverSeqNum[clientIndexToSend]++;
        currentOffset += size;
    }

    OutputDebugString(L"\n\n= SendTextureUdp: Sent texture =========");
}

bool ServerNetworkManager::RecvCameraDataUdp(
    std::vector<std::array<float3, 3>>& cameraData,
    std::vector<std::mutex>& mutexCameraData,
    SOCKET& socketUdp,
    bool useLongTimeout)
{
    // Assumes server is receiving cam data from client
    UdpCustomPacketHeader recvHeader;
    bool hasReceived;
    char* packetData;
    char* recvBuffer = new char[DEFAULT_BUFLEN];
    int clientIndex;
    if (useLongTimeout)
    {
        hasReceived = RecvUdpCustom(recvBuffer, recvHeader, packetData, socketUdp,
            clientIndex, UDP_FIRST_TIMEOUT_MS);
    }
    else
    {
        hasReceived = RecvUdpCustom(recvBuffer, recvHeader, packetData, socketUdp, clientIndex);
    }

    if (!hasReceived)
    {
        delete[] recvBuffer;
        char bufferSn[75];
        sprintf(bufferSn, "\n\n= RecvCameraDataUdp: Failed to receive %d =========", clientSeqNum[clientIndex]);
        OutputDebugStringA(bufferSn);
        if (cameraDataCache.empty())
        {   
            OutputDebugString(L"\n= Camera data cache empty =========");
            // Fail, nothing in cache
            return false;
        }
        else
        {
            OutputDebugString(L"\n= Using old camera data value =========");
            // Take from the cache
            cameraData[clientIndex] = cameraDataCache.back();
            clientSeqNum[clientIndex]++;
            return true;
        }
    }
    else
    {
        // if new client, initialise cam data 
        if (clientIndex == -1) {
            clientIndex = (int)mClientAddresses.size() - 1; // actual new client index is the last one added
            cameraData.push_back({ { {0, 0, 0}, {0, 0, 0}, {0, 0, 0} } });
        }
        // Increment sequence number for next communication
        clientSeqNum[clientIndex]++;
        {
            assert(recvHeader.dataSize == sizeof(cameraData[clientIndex]));
            uint8_t* dataOut = reinterpret_cast<uint8_t*>(&cameraData[clientIndex]);
            
            // Copy the data to the pointer
            std::lock_guard<std::mutex> lock(mutexCameraData[clientIndex]);
            for (int i = 0; i < recvHeader.dataSize; i++)
            {
                dataOut[i] = packetData[i];
            }
        }
        delete[] recvBuffer;

        mClientCamPosUpdated[clientIndex].signal();
        mSpServerCamPosUpdated.signal();

        // Populate the cache
        cameraDataCache.push_back(cameraData[clientIndex]);
        if (cameraDataCache.size() > maxCamDataCacheSize)
        {
            cameraDataCache.pop_front();
        }
        return true;
    }
}

bool ServerNetworkManager::RecvUdpCustom(
    char* dataBuffer,
    UdpCustomPacketHeader& outDataHeader,
    char*& outDataPointer,
    SOCKET& socketUdp,
    int &fromClientIndex, // OUT
    int timeout)
{
    int headerSize = UdpCustomPacket::headerSizeBytes;
    int dataReceivedSoFar = 0;

    // Set timeout for the socket
    if (setsockopt(socketUdp, SOL_SOCKET, SO_RCVTIMEO, reinterpret_cast<char*>(&timeout), sizeof(int)) != 0)
    {
        char buffer[61];
        sprintf(buffer, "Set socket options failed with error code: %d", WSAGetLastError());
        OutputDebugStringA(buffer);
        return false;
    }

    struct sockaddr_in clientAddr;
    struct sockaddr* clientAddrPtr = reinterpret_cast<struct sockaddr*>(&clientAddr);
    int addrLen = sizeof(*clientAddrPtr);
    // Read header for packet size
    do
    {
        int iResult = recvfrom(socketUdp, &(dataBuffer[dataReceivedSoFar]),
                               DEFAULT_BUFLEN, 0, clientAddrPtr, &addrLen);
        if (iResult != SOCKET_ERROR)
        {
            dataReceivedSoFar += iResult;
        }
        else
        {
            int lastError = WSAGetLastError();
            if (lastError == WSAETIMEDOUT)
            {
                OutputDebugStringA(
                    "\nRecvUdpCustom: Error receiving, connection timed out.");
            }
            else
            {
                char buffer[58];
                sprintf(buffer,
                    "\nRecvUdpCustom: Error receiving: %d", lastError);
                OutputDebugStringA(buffer);
            }
            return false;
        }
    } while (dataReceivedSoFar < headerSize);

    // Update data header
    outDataHeader = UdpCustomPacket::getHeader(dataBuffer);
    // Update data pointer
    outDataPointer = dataBuffer + headerSize;


    // NEW CLIENT
    if (!mapClientAddressToIndex.count(clientAddr.sin_addr.S_un.S_addr))
    {
        fromClientIndex = -1; // new client
        // add to list of client addresses
        mClientAddresses.push_back(clientAddr);
        clientSeqNum.push_back(1); // this particular clients seq num
        serverSeqNum.push_back(1); // server seq num for this particular client

        // add to the maps of client address to index
        mapClientAddressToIndex.insert(std::pair{ clientAddr.sin_addr.S_un.S_addr, (int)(mClientAddresses.size() - 1)});
    }
    else {
        fromClientIndex = mapClientAddressToIndex[clientAddr.sin_addr.S_un.S_addr];
    }

    int totalPacketSize = outDataHeader.dataSize + headerSize;
    // Receive the rest of the packet, if needed
    while (dataReceivedSoFar < totalPacketSize)
    {
        int iResult = recvfrom(socketUdp, &(dataBuffer[dataReceivedSoFar]),
                                DEFAULT_BUFLEN, 0, clientAddrPtr, &addrLen);
        if (iResult != SOCKET_ERROR)
        {
            dataReceivedSoFar += iResult;
        }
        else
        {
            int lastError = WSAGetLastError();
            if (lastError == WSAETIMEDOUT)
            {
                OutputDebugStringA(
                    "\nRecvUdpCustom: "
                    "Error receiving rest of packet, connection timed out.");
            }
            else
            {
                char buffer[58];
                sprintf(buffer,
                    "\nRecvUdpCustom: "
                    "Error receiving rest of packet: %d", lastError);
                OutputDebugStringA(buffer);
            }
            return false;
        }
    }

    if (dataReceivedSoFar > totalPacketSize)
    {
        char extraDataBuffer[85];
        sprintf(extraDataBuffer, "\nRecvUdpCustom: Ignoring extra %d bytes for packet #%d",
                dataReceivedSoFar - totalPacketSize, outDataHeader.sequenceNumber);
        OutputDebugStringA(extraDataBuffer);
    }

    return true;
}

bool ServerNetworkManager::RecvUdpCustomAndCheck(
    char* dataBuffer,
    UdpCustomPacketHeader& outDataHeader,
    char*& outDataPointer,
    SOCKET& socketUdp,
    int expectedSeqNum,
    int timeout)
{
    if (!RecvUdpCustom(dataBuffer, outDataHeader, outDataPointer, socketUdp, timeout))
    {
        return false;
    }

    // Check the sequence number
    int recvSeqNum = outDataHeader.sequenceNumber;
    if (recvSeqNum != expectedSeqNum)
    {
        char buffer[88];
        sprintf(buffer, "\nSequence number does not match, expected %d, received %d",
                        expectedSeqNum, recvSeqNum);
        OutputDebugStringA(buffer);
        return false;
    }
    else
    {
        return true;
    }
}

bool ServerNetworkManager::SendUdpCustom(UdpCustomPacketHeader& dataHeader, char* dataToSend, int clientIndex, SOCKET& socketUdp)
{
    std::unique_ptr<char[]> udpToSend = dataHeader.createUdpPacket(dataToSend);

    // Send the data
    char msgBuffer0[65];
    sprintf(msgBuffer0, "\n\n= SendUdpCustom: Sending packet %d... =========", dataHeader.sequenceNumber);
    OutputDebugStringA(msgBuffer0);

    struct sockaddr* toSocket = reinterpret_cast<sockaddr*>(&mClientAddresses[clientIndex]);
    int socketLen = sizeof(mClientAddresses[clientIndex]);
    int sendSize = UdpCustomPacket::headerSizeBytes + dataHeader.dataSize;
    int sentSoFar = 0;

    while (sentSoFar < sendSize)
    {
        int iResult = sendto(socketUdp, &(udpToSend[sentSoFar]), sendSize, 0, toSocket, socketLen);
        if (iResult != SOCKET_ERROR)
        {
            sentSoFar += iResult;
        }
        else
        {
            char msgBuffer1[61];
            sprintf(msgBuffer1, "\n\n= SendUdpCustom: Socket error, %d =========", WSAGetLastError());
            OutputDebugStringA(msgBuffer1);

            char msgBuffer2[89];
            sprintf(msgBuffer2, "\n= SendUdpCustom: Failed to send packet with sequence number %d =========",
                    dataHeader.sequenceNumber);
            OutputDebugStringA(msgBuffer2);

            return false;
        }
    }
    
    OutputDebugString(L"\n\n= SendUdpCustom: Sent packets =========");
    return true;
}
