#include "../NetworkPasses/NetworkClientRecvPass.h"

#include "ClientNetworkManager.h"


// UDP Client
Semaphore ClientNetworkManager::mSpClientCamPosReadyToSend(false);
Semaphore ClientNetworkManager::mSpClientSeqTexRecv(false);
std::mutex ClientNetworkManager::mMutexClientVisTexRead;

bool ClientNetworkManager::SetUpClientUdp(PCSTR serverName, PCSTR serverPort)
{
    int slen = sizeof(serverAddress);
    WSADATA wsa;

    //Initialise winsock
    printf("\nInitialising Winsock...");
    if (WSAStartup(MAKEWORD(2, 2), &wsa) != 0)
    {
        printf("Failed. Error Code : %d", WSAGetLastError());
        exit(EXIT_FAILURE);
    }
    printf("Initialised.\n");

    //create socket
    if ((mClientUdpSock = socket(AF_INET, SOCK_DGRAM, IPPROTO_UDP)) == SOCKET_ERROR)
    {
        printf("socket() failed with error code : %d", WSAGetLastError());
        exit(EXIT_FAILURE);
    }

    //setup address structure
    memset((char*)&serverAddress, 0, sizeof(serverAddress));
    serverAddress.sin_family = AF_INET;
    serverAddress.sin_port = htons((u_short)std::strtoul(serverPort, NULL, 0));
    inet_pton(AF_INET, serverName, &serverAddress.sin_addr.S_un.S_addr);

    //bind
    //if (bind(mClientUdpSock, (struct sockaddr*) &serverAddress, sizeof(serverAddress)) == SOCKET_ERROR)
    //{
    //    char buffer[69];
    //    sprintf(buffer, "\n\n= Pre-Falcor Init - Bind failed with error code: %d", WSAGetLastError());
    //    OutputDebugStringA(buffer);
    //    exit(EXIT_FAILURE);
    //}
    
    //mSi_otherUdp.sin_addr.S_un.S_addr = inet_addr(serverName);
    return true;
}

void ClientNetworkManager::ListenClientUdp(bool isFirstReceive, bool executeForever)
{
    int32_t latestFrameRecv = 0;

    while (true)
    {
        std::chrono::time_point startOfFrame = std::chrono::system_clock::now();

        // Await server to send back the visibility pass texture
        OutputDebugString(L"\n\n= Awaiting visTex receiving over network... =========");
        int visTexLen = VIS_TEX_LEN;
        FrameData rcvdFrameData = { visTexLen, latestFrameRecv, 0 };
        
        char* toRecvData = NetworkClientRecvPass::clientWriteBuffer;
        int recvStatus;

        // Store the time when the first frame was received
        // (server sends timestamps relative to the time when the first frame was fully rendered)
        if (startTime == std::chrono::milliseconds::zero())
        {
            startTime = getCurrentTime();
        }

        recvStatus = RecvTextureUdp(rcvdFrameData, toRecvData, mClientUdpSock);

        std::chrono::milliseconds currentTime = getComparisonTimestamp();
        std::chrono::milliseconds timeDifference = currentTime - std::chrono::milliseconds(rcvdFrameData.timestamp);
        if (timeDifference > std::chrono::milliseconds::zero())
        {
            char slowerMessage[77];
            sprintf(slowerMessage, "\n=Client received texture %d ms slower than expected=========",
                    static_cast<int>(timeDifference.count()));
            OutputDebugStringA(slowerMessage);
        }
        else
        {
            char fasterMessage[77];
            sprintf(fasterMessage, "\n=Client received texture %d ms faster than expected=========",
                    static_cast<int>(timeDifference.count()));
            OutputDebugStringA(fasterMessage);
        }
        
        if (recvStatus == 0)
        {
            char frameDataMessage[90];
            sprintf(frameDataMessage, "\n\n= Discarding frame %d (size: %d, time: %d)\n",
                    rcvdFrameData.frameNumber, rcvdFrameData.frameSize, rcvdFrameData.timestamp);
            OutputDebugStringA(frameDataMessage);
        }
        else // recvStatus == 1
        {
            OutputDebugString(L"\n\n= visTex received over network =========");
            char frameDataMessage[89];
            sprintf(frameDataMessage, "\nFrameData: Number: %d, Size: %d, Time: %d\n",
                    rcvdFrameData.frameNumber, rcvdFrameData.frameSize, rcvdFrameData.timestamp);
            OutputDebugStringA(frameDataMessage);

            // find the difference in frame number for prediction
            numFramesBehind = clientFrameNum - rcvdFrameData.frameNumber;

            if (numFramesBehind == 0) {
                mSpClientSeqTexRecv.signal();
            }

            if (compression) {
                int decompressedSize = Compression::executeLZ4Decompress(NetworkClientRecvPass::clientWriteBuffer, 
                    NetworkClientRecvPass::intermediateBuffer, rcvdFrameData.frameSize, VIS_TEX_LEN);
                char* tempPtr = NetworkClientRecvPass::intermediateBuffer;
                NetworkClientRecvPass::intermediateBuffer = NetworkClientRecvPass::clientWriteBuffer;
                NetworkClientRecvPass::clientWriteBuffer = tempPtr;
            }
            else {
                mOutputBufferSize = rcvdFrameData.frameSize;
            }

            // acquire reading buffer mutex to swap buffers
            {
                std::lock_guard readingLock(ClientNetworkManager::mMutexClientVisTexRead);
                char* tempPtr = NetworkClientRecvPass::clientReadBuffer;
                NetworkClientRecvPass::clientReadBuffer = NetworkClientRecvPass::clientWriteBuffer;
                NetworkClientRecvPass::clientWriteBuffer = tempPtr;

                mOutputBuffer = NetworkClientRecvPass::clientReadBuffer;
                // mutex and lock are released at the end of scope
            }

            std::chrono::time_point endOfFrame = std::chrono::system_clock::now();
            std::chrono::duration<double> diff = endOfFrame - startOfFrame;
            char printFps[102];
            sprintf(printFps, "\n\n= ListenClientUdp - Frame took %.10f s, estimated FPS: %.2f =========", diff.count(), getFps(diff));
            OutputDebugStringA(printFps);

            // measuring time from camera sent to texture received for a given frame
            updateTimeForFrame(rcvdFrameData.frameNumber, endOfFrame);
        }

        if (!executeForever)
        {
            break;
        }
    }
}

void ClientNetworkManager::SendWhenReadyClientUdp(Scene::SharedPtr mpScene)
{
    while (true)
    {
        std::chrono::time_point startOfFrame = std::chrono::system_clock::now();
        mSpClientCamPosReadyToSend.wait();

        // store cameraU, V, W specifically for GBuffer rendering later
        Camera::SharedPtr cam = mpScene->getCamera();
        const CameraData& cameraData = cam->getData();
        cameraUX = cameraData.cameraU.x;
        cameraUY = cameraData.cameraU.y;
        cameraUZ = cameraData.cameraU.z;
        cameraVX = cameraData.cameraV.x;
        cameraVY = cameraData.cameraV.y;
        cameraVZ = cameraData.cameraV.z;
        cameraWX = cameraData.cameraW.x;
        cameraWY = cameraData.cameraW.y;
        cameraWZ = cameraData.cameraW.z;

        // Increment client frame number
        clientFrameNum++;
        
        OutputDebugString(L"\n\n= Awaiting camData sending over network... =========");
        SendCameraDataUdp(cam, mClientUdpSock);
        OutputDebugString(L"\n\n= camData sent over network =========");
        
        // store this time as camera sent
        int32_t currentClientFrameNum = clientFrameNum;
        auto storeTime = std::make_pair(currentClientFrameNum, std::chrono::system_clock::now());
        timeAtCameraSent.emplace(storeTime);
    }   
}

bool ClientNetworkManager::CloseClientConnectionUdp()
{
    closesocket(mClientUdpSock);
    WSACleanup();
    return true;
}

double ClientNetworkManager::getTimeForOneSequentialFrame()
{
    double res = timeForOneSequentialFrame.count() * 1000;
    return res;
}

int ClientNetworkManager::RecvTextureUdp(FrameData& frameDataOut, char* outRecvTexData, SOCKET& socketUdp, int timeout)
{
    int recvTexSize = frameDataOut.frameSize;
    int latestFrameRecv = frameDataOut.frameNumber;
    // Initialise numberOfPackets to the expected number without compression
    int numberOfPackets = recvTexSize / UdpCustomPacket::maxPacketSize +
        ((recvTexSize % UdpCustomPacket::maxPacketSize > 0) ? 1 : 0);

    int receivedDataSoFar = 0;
    uint8_t* dataPtr = reinterpret_cast<uint8_t*>(outRecvTexData);
    char oldHeaderBytes[UdpCustomPacket::headerSizeBytes];

    int relativeSeqNum = 0;
    int numOfRecvAttempts = 0;
    while (relativeSeqNum < numberOfPackets)
    {
        // Total offset of the pointer from the start for this packet
        int offset = UdpCustomPacket::maxPacketSize * relativeSeqNum;
        char* recvPacketStart = &outRecvTexData[offset - UdpCustomPacket::headerSizeBytes];

        // Copy old bytes that will be overwritten by packet header
        if (relativeSeqNum > 0)
        {
            for (int j = 0; j < UdpCustomPacket::headerSizeBytes; j++)
            {
                oldHeaderBytes[j] = recvPacketStart[j];
            }
        }

        UdpCustomPacketHeader recvHeader;
        char* dataPointer;
        // CASE 0: RECV FAILS
        if (!RecvUdpCustom(recvPacketStart, recvHeader, dataPointer, socketUdp, timeout))
        {
            char buffer[73];
            sprintf(buffer, "\n\n= RecvTextureUdp: Failed to receive packet %d =========", serverSeqNum);
            OutputDebugStringA(buffer);
            continue;
        }
        // CASE 1: FIRST PACKET
        if (relativeSeqNum == 0) {
            // CASE 1.1: OLD FRAME
            // skip past packets that belong to an older frame
            if (recvHeader.frameNumber <= latestFrameRecv)
            {
                char buffer[93];
                sprintf(buffer, "\n\n= RecvTextureUdp: "
                    "Skipping past packet %d for older frame %d",
                    recvHeader.sequenceNumber, recvHeader.frameNumber);
                OutputDebugStringA(buffer);
                continue;
            }
            // CASE 1.2: NEW FRAME
            // Remember frame data for the first full packet received
            numberOfPackets = recvHeader.numOfFramePackets;
            frameDataOut.timestamp = recvHeader.timestamp;
            serverSeqNum = recvHeader.sequenceNumber;
            latestFrameRecv = recvHeader.frameNumber;
        }
        // CASE 2: NOT FIRST PACKET
        else {
            // CASE 2.1: RECEIVED OLDER PACKET
            if (recvHeader.sequenceNumber < serverSeqNum) {
                char buffer[93];
                sprintf(buffer, "\n\n= RecvTextureUdp: "
                    "Skipping past packet %d for older frame %d",
                    recvHeader.sequenceNumber, recvHeader.frameNumber);
                OutputDebugStringA(buffer);

                // copy back
                for (int j = 0; j < UdpCustomPacket::headerSizeBytes; j++)
                {
                    recvPacketStart[j] = oldHeaderBytes[j];
                }
                continue;
            }
            // CASE 2.2: RECEIVED FUTURE PACKET
            // reset latest frame received, return 0 (error)
            else if (recvHeader.sequenceNumber > serverSeqNum)
            {
                char buffer[180];
                sprintf(buffer, "\n\n= RecvTextureUdp: "
                    "Sequence number does not match, expected %d, received %d\n"
                    "Setting expected latest frame num to %d",
                    serverSeqNum, recvHeader.sequenceNumber, recvHeader.frameNumber);
                OutputDebugStringA(buffer);

                frameDataOut.frameNumber = recvHeader.frameNumber;
                // Reject the entire frame and return
                return 0;
            }
            // CASE 2.3 SUCCESS, CORRECT PACKET RECV
            // Replace the original bytes at the header position, now that we have the header data            
            for (int j = 0; j < UdpCustomPacket::headerSizeBytes; j++)
            {
                recvPacketStart[j] = oldHeaderBytes[j];
            }

            // Increment for the next packet
            relativeSeqNum++;
            serverSeqNum++;
            receivedDataSoFar += recvHeader.dataSize;
        }
    }
    frameDataOut.frameSize = receivedDataSoFar;
    OutputDebugString(L"\n\n= RecvTextureUdp: Received texture =========");

    return 1;
}

bool ClientNetworkManager::SendCameraDataUdp(Camera::SharedPtr camera, SOCKET& socketUdp)
{
    std::array<float3, 3> cameraData = { camera->getPosition(), camera->getUpVector(), camera->getTarget() };
    char* data = reinterpret_cast<char*>(&cameraData);
    // Assumes client sending to server
    UdpCustomPacketHeader headerToSend(clientSeqNum, sizeof(cameraData), clientFrameNum);
    
    clientSeqNum++;
    
    bool wasDataSent = true;
    if (!SendUdpCustom(headerToSend, data, socketUdp))
    {
        OutputDebugString(L"\n\n= SendCameraDataUdp: Failed to send =========");
        wasDataSent = false;
    }
    return wasDataSent;
}

bool ClientNetworkManager::RecvUdpCustom(
    char* dataBuffer,
    UdpCustomPacketHeader& outDataHeader,
    char*& outDataPointer,
    SOCKET& socketUdp,
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

bool ClientNetworkManager::SendUdpCustom(UdpCustomPacketHeader& dataHeader, char* dataToSend, SOCKET& socketUdp)
{
    std::unique_ptr<char[]> udpToSend = dataHeader.createUdpPacket(dataToSend);

    // Send the data
    char msgBuffer0[65];
    sprintf(msgBuffer0, "\n\n= SendUdpCustom: Sending packet %d... =========", dataHeader.sequenceNumber);
    OutputDebugStringA(msgBuffer0);

    struct sockaddr* toSocket = reinterpret_cast<sockaddr*>(&serverAddress);
    int socketLen = sizeof(serverAddress);
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

std::chrono::milliseconds ClientNetworkManager::getComparisonTimestamp()
{
    return getCurrentTime() - startTime;
}

inline void ClientNetworkManager::updateTimeForFrame(int frameReceived,
    std::chrono::time_point<std::chrono::system_clock> endOfFrame)
{
    auto seqTime = timeAtCameraSent.front();
    
    while (frameReceived > seqTime.first) {
        timeAtCameraSent.pop();
        seqTime = timeAtCameraSent.front();
    };

    char msgBuffer1[61];
    sprintf(msgBuffer1, "\n\n= frame received %d, seq time first : %d =========", frameReceived, seqTime.first);
    OutputDebugStringA(msgBuffer1);

    timeForOneSequentialFrame = endOfFrame - seqTime.second;
    timeAtCameraSent.pop();
}
