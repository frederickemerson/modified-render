#include "../NetworkPasses/NetworkServerRecvPass.h"
#include "../NetworkPasses/NetworkServerSendPass.h"

#include "ServerNetworkManager.h"

// UDP Server
std::array<Semaphore, MAX_NUM_CLIENT> ServerNetworkManager::mClientCamPosUpdated = {Semaphore(false),
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

bool ServerNetworkManager::ListenServerUdp()
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
                          mServerUdpSock);
        OutputDebugString(L"\n\n= NetworkThread - camData received over network =========");

        std::chrono::time_point endOfFrame = std::chrono::system_clock::now();
        std::chrono::duration<double> diff = endOfFrame - startOfFrame;
        char printFps[102];
        sprintf(printFps, "\n\n= ListenServerUdp - Frame took %.10f s, estimated FPS: %.2f =========", diff.count(), getFps(diff));
        OutputDebugStringA(printFps);
    }
    while (true);

    return true;
}

void ServerNetworkManager::SendWhenReadyServerUdp(
    RenderContext* pRenderContext,
    std::shared_ptr<ResourceManager> pResManager,
    int texWidth,
    int texHeight)
{
    // Keep track of time
    int numFramesRendered = 0;
    std::chrono::milliseconds timeOfFirstFrame;

    while (true)
    {
        std::chrono::time_point startOfFrame = std::chrono::system_clock::now();
        // Allow rendering using the camPos to begin, and wait for visTex to complete rendering
        OutputDebugString(L"\n\n= NetworkThread - Awaiting visTex to finish rendering... =========");
        mSpServerVisTexComplete.wait();
        OutputDebugString(L"\n\n= NetworkThread - VisTex finished rendering. Awaiting visTex sending over network... =========");

        //-------------client index --------------
        int clientIndex = 0;
        clientIndex = sendClientQueue.front();
        sendClientQueue.pop();

        //----------client frame num-------------
        int frameNum = NetworkServerRecvPass::frameNumRendered.front();
        NetworkServerRecvPass::frameNumRendered.pop();

        std::string frameMsg = std::string("\n\n====== FRAME ") + std::to_string(frameNum)
            + std::string(" OF CLIENT ") + std::to_string(clientIndex) + std::string("==========");
        OutputDebugString(string_2_wstring(frameMsg).c_str());

        {
            std::lock_guard lock(mMutexServerVisTexRead);
            char* toSendData = mGetInputBuffer();
            // The size of the actual Buffer
            // that is given by Falcor is less then VIS_TEX_LEN
            // 
            // The actual size is the screen width and height * 4
            // We send VIS_TEX_LEN but we need to compress with the actual
            // size to prevent reading outside of the Falcor Buffer
            int toSendSize = mGetInputBufferSize();

            if (compression) {
                int compressedSize = Compression::executeLZ4Compress(mGetInputBuffer(), 
                    NetworkServerSendPass::intermediateBuffer, mGetInputBufferSize());
                toSendData = NetworkServerSendPass::intermediateBuffer;
                toSendSize = compressedSize;
                //std::string frameMsg = std::string("\n\nCompressed texture to size: ") + std::to_string(compressedSize);
                //OutputDebugString(string_2_wstring(frameMsg).c_str());
            }

            // Send the visBuffer back to the sender
            // Generate timestamp
            std::chrono::milliseconds currentTime = getCurrentTime();
            int timestamp = static_cast<int>((currentTime - timeOfFirstFrame).count());

            std::thread{ &ServerNetworkManager::SendTextureUdp, this, FrameData{ toSendSize, frameNum, timestamp },
                           toSendData,
                           clientIndex,
                           mServerUdpSock }.detach();
            /*
            SendTextureUdp({ toSendSize, frameNum, timestamp },
                           toSendData,
                           clientIndex,
                           mServerUdpSock);
            */
        }
        
        // increase local count of frames rendered by server
        numFramesRendered++;

        // output end message
        OutputDebugString(L"\n\n= NetworkThread - visTex sent over network =========");
        std::string endMsg = std::string("\n\n================================ Frame ") +
            std::to_string(frameNum) + std::string(" COMPLETE for client #") +
            std::to_string(clientIndex) + std::string(", ") + std::to_string(numFramesRendered) +
            (" total frames rendered by server ================================");
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

void ServerNetworkManager::setArtificialLag(int milliseconds) {
    artificialLag = std::chrono::milliseconds(milliseconds);
}

void ServerNetworkManager::SendTextureUdp(FrameData frameData, char* sendTexData, int clientIndex, const SOCKET& socketUdp)
{
    std::this_thread::sleep_for(artificialLag);
    // Variable splitSize controls the size of the split packets
    int32_t splitSize = UdpCustomPacket::maxPacketSize;
    int16_t numOfFramePackets = frameData.frameSize / splitSize +
                                ((frameData.frameSize % splitSize > 0) ? 1 : 0);
    
    // Split the frame data and send
    int currentOffset = 0;
    bool isFirst = true;

    for (int32_t amountLeft = frameData.frameSize; amountLeft > 0; amountLeft -= splitSize)
    {
        int32_t size = std::min(amountLeft, UdpCustomPacket::maxPacketSize);
        UdpCustomPacketHeader texHeader(serverSeqNum[clientIndex], size, frameData.frameNumber,
                                        numOfFramePackets, frameData.timestamp, isFirst);
        isFirst = false;

        if (!SendUdpCustom(texHeader, &sendTexData[currentOffset], clientIndex, socketUdp))
        {
            char buffer[70];
            sprintf(buffer, "\n\n= SendTextureUdp: Failed to send packet %d =========",
                    texHeader.sequenceNumber);
            OutputDebugStringA(buffer);
            return;
        }

        serverSeqNum[clientIndex]++;
        currentOffset += size;
    }

    OutputDebugString(L"\n\n= SendTextureUdp: Sent texture =========");
}

bool ServerNetworkManager::RecvCameraDataUdp(
    std::vector<std::array<float3, 3>>& cameraData,
    std::array<std::mutex, MAX_NUM_CLIENT>& mutexCameraData,
    SOCKET& socketUdp)
{
    // Assumes server is receiving cam data from client
    UdpCustomPacketHeader recvHeader;
    bool hasReceived;
    char* packetData;
    char* recvBuffer = new char[DEFAULT_BUFLEN];
    int clientIndex = 0; // recv udp custom will set this value

    hasReceived = RecvUdpCustom(recvBuffer, recvHeader, packetData, socketUdp, clientIndex);

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

        // Get frame number from the packet
        clientFrameNum[clientIndex] = recvHeader.frameNumber;

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
        clientFrameNum.push_back(0); // this clients frame number

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

bool ServerNetworkManager::SendUdpCustom(const UdpCustomPacketHeader& dataHeader, char* dataToSend, int clientIndex, const SOCKET& socketUdp)
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
