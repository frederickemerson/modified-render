#include "ResourceManager.h"
#include "../NetworkPasses/NetworkPass.h"
#include "../NetworkPasses/NetworkUtils.h"
#include "NetworkManager.h"

#include "lz4.h"

bool NetworkManager::mCamPosReceivedTcp = false;
bool NetworkManager::mVisTexCompleteTcp = false;
bool NetworkManager::mCompression = true;

std::vector<char> NetworkManager::wrkmem(LZO1X_1_MEM_COMPRESS, 0);
std::vector<unsigned char> NetworkManager::compData(OUT_LEN(POS_TEX_LEN), 0);

// UDP Client
Semaphore NetworkManager::mSpClientCamPosReadyToSend(false);
std::mutex NetworkManager::mMutexClientVisTexRead;

// UDP Server
Semaphore NetworkManager::mSpServerVisTexComplete(false);
Semaphore NetworkManager::mSpServerCamPosUpdated(false);
std::mutex NetworkManager::mMutexServerVisTexRead;
std::mutex NetworkManager::mMutexServerCamData;

// TCP
std::mutex NetworkManager::mMutexServerVisTexTcp;
std::condition_variable NetworkManager::mCvCamPosReceived;
std::condition_variable NetworkManager::mCvVisTexComplete;

/// <summary>
/// Initialise server side connection, opens up a TCP listening socket at given port
/// and waits for client to connect to the socket. This call is blocking until a client
/// successfully connects with the socket. On connecting, client will send 2 integers which
/// are their texture width and texture height to be used by server in subsequent rendering frames.
/// </summary>
/// <param name="port">- port number, prefrabbly >1024</param>
/// <param name="outTexWidth">- variable to receive client texture width</param>
/// <param name="outTexHeight">- variable to receive client texture height</param>
/// <returns></returns>
bool NetworkManager::SetUpServer(PCSTR port, int& outTexWidth, int& outTexHeight)
{
    WSADATA wsaData;
    int iResult;

    mListenSocket = INVALID_SOCKET;
    mClientSocket = INVALID_SOCKET;

    struct addrinfo* result = NULL;
    struct addrinfo hints;

    OutputDebugString(L"\n\n= Pre-Falcor Init - NetworkManager::SetUpServer - PIPELINE SERVER SETTING UP =========");

    // Initialize Winsock
    iResult = WSAStartup(MAKEWORD(2, 2), &wsaData);
    if (iResult != 0) {
        std::string errString = std::string("\n\n= Pre-Falcor Init - WSAStartup failed with error: ") + std::to_string(iResult);
        OutputDebugString(string_2_wstring(errString).c_str());
        return false;
    }

    ZeroMemory(&hints, sizeof(hints));
    hints.ai_family = AF_INET;
    hints.ai_socktype = SOCK_STREAM;
    hints.ai_protocol = IPPROTO_TCP;
    hints.ai_flags = AI_PASSIVE;

    // Resolve the server address and port
    iResult = getaddrinfo(NULL, port, &hints, &result);
    if (iResult != 0) {
        std::string errString = std::string("\n\n= Pre-Falcor Init - getaddrinfo failed with error: ") + std::to_string(iResult);
        OutputDebugString(string_2_wstring(errString).c_str());
        WSACleanup();
        return false;
    }

    // Create a SOCKET for connecting to server
    mListenSocket = socket(result->ai_family, result->ai_socktype, result->ai_protocol);
    if (mListenSocket == INVALID_SOCKET) {
        std::string errString = std::string("\n\n= Pre-Falcor Init - socket failed with error: ") + std::to_string(WSAGetLastError());
        OutputDebugString(string_2_wstring(errString).c_str());
        freeaddrinfo(result);
        WSACleanup();
        return false;
    }

    // Setup the TCP listening socket
    iResult = bind(mListenSocket, result->ai_addr, (int)result->ai_addrlen);
    if (iResult == SOCKET_ERROR) {
        std::string errString = std::string("\n\n= Pre-Falcor Init - bind failed with error: ") + std::to_string(WSAGetLastError());
        OutputDebugString(string_2_wstring(errString).c_str());
        freeaddrinfo(result);
        closesocket(mListenSocket);
        WSACleanup();
        return false;
    }

    freeaddrinfo(result);

    OutputDebugString(L"\n\n= Pre-Falcor Init - SETUP COMPLETE =========");

    // Listening for client socket
    OutputDebugString(L"\n\n= Pre-Falcor Init - Trying to listen for client... =========");
    iResult = listen(mListenSocket, SOMAXCONN);
    if (iResult == SOCKET_ERROR)
    {
        std::string errString = std::string("\n\n= Pre-Falcor Init - listen failed with error: ") + std::to_string(WSAGetLastError());
        OutputDebugString(string_2_wstring(errString).c_str());
        closesocket(mListenSocket);
        WSACleanup();
        return false;
    }
    // Accept the client socket
    OutputDebugString(L"\n\n= Pre-Falcor Init - Trying to accept client... =========");
    mClientSocket = accept(mListenSocket, NULL, NULL);
    if (mClientSocket == INVALID_SOCKET)
    {
        std::string errString = std::string("\n\n= Pre-Falcor Init - accept failed with error: ") + std::to_string(WSAGetLastError());
        OutputDebugString(string_2_wstring(errString).c_str());
        closesocket(mListenSocket);
        WSACleanup();
        return false;
    }
    // No longer need server socket
    closesocket(mListenSocket);
    OutputDebugString(L"\n\n= Pre-Falcor Init - Connection with client established =========");

    // Get the client texture width/height
    OutputDebugString(L"\n\n= Pre-Falcor Init - Getting client texture width/height... =========");
    RecvInt(outTexWidth, mClientSocket);
    RecvInt(outTexHeight, mClientSocket);
    OutputDebugString(L"\n\n= Pre-Falcor Init - Texture width/height received =========");

    return true;
}

bool NetworkManager::SetUpServerUdp(PCSTR port, int& outTexWidth, int& outTexHeight)
{
    WSADATA wsa;

    //Initialise winsock
    OutputDebugString(L"\n\n= Pre-Falcor Init - NetworkManager::SetUpServerUdp - Initialising Winsock... =========");
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

    // Listening for client socket
    OutputDebugString(L"\n\n= Pre-Falcor Init - Trying to listen for client width/height... =========");

    UdpCustomPacketHeader firstPacketHeader;
    char* dataPointer;
    char* recvBuffer = new char[DEFAULT_BUFLEN];
    // Get the client texture width/height    
    // Expected sequence number is 0
    if (!RecvUdpCustomAndCheck(recvBuffer, firstPacketHeader, dataPointer,
                               mServerUdpSock, 0, UDP_FIRST_TIMEOUT_MS, true))
    {
        OutputDebugString(L"\n\n= Pre-Falcor Init - FAILED to receive UDP packet from client =========");
        return false;
    }
    // Next client sequence number should be 1
    clientSeqNum = 1;
    
    // Packet should consist of two ints
    if (firstPacketHeader.dataSize != 8)
    {
        OutputDebugString(L"\n\n= Pre-Falcor Init - FAILED: UDP packet from client has wrong size =========");
        return false;
    }

    int* widthAndHeight = reinterpret_cast<int*>(dataPointer);
    outTexWidth = widthAndHeight[0];
    outTexHeight = widthAndHeight[1];
    delete[] recvBuffer;

    OutputDebugString(L"\n\n= Pre-Falcor Init - Texture width/height received =========");
    char printWidthHeight[52];
    sprintf(printWidthHeight, "\nWidth: %d\nHeight: %d", outTexWidth, outTexHeight);
    OutputDebugStringA(printWidthHeight);

    return true;
}

bool NetworkManager::ListenServerUdp(bool executeForever, bool useLongTimeout)
{
    // Receive until the peer shuts down the connection
    do
    {
        std::chrono::time_point startOfFrame = std::chrono::system_clock::now();
        // Receive the camera position from the sender
        OutputDebugString(L"\n\n= NetworkThread - Awaiting camData receiving over network... =========");
        // Mutex will be locked in RecvCameraDataUdp
        RecvCameraDataUdp(NetworkPass::camData,
                          NetworkManager::mMutexServerCamData,
                          mServerUdpSock,
                          useLongTimeout);
        OutputDebugString(L"\n\n= NetworkThread - camData received over network =========");
        mSpServerCamPosUpdated.signal();

        std::chrono::time_point endOfFrame = std::chrono::system_clock::now();
        std::chrono::duration<double> diff = endOfFrame - startOfFrame;
        char printFps[102];
        sprintf(printFps, "\n\n= ListenServerUdp - Frame took %.10f s, estimated FPS: %.2f =========", diff.count(), getFps(diff));
        OutputDebugStringA(printFps);
    }
    while (executeForever);

    return true;
}

void NetworkManager::SendWhenReadyServerUdp(
    RenderContext* pRenderContext,
    std::shared_ptr<ResourceManager> pResManager,
    int texWidth,
    int texHeight)
{
    int numFramesRendered = 0;
    // Keep track of time
    std::chrono::milliseconds timeOfFirstFrame;
    // for compression
    std::unique_ptr<char[]> compressionBuffer = std::make_unique<char[]>(VIS_TEX_LEN);

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
            char* toSendData = (char*)NetworkPass::pVisibilityDataServer;

            // The size of the actual Buffer (pointed to by pVisibilityDataServer)
            // that is given by Falcor is less then VIS_TEX_LEN
            // 
            // The actual size is the screen width and height * 4
            // We send VIS_TEX_LEN but we need to compress with the actual
            // size to prevent reading outside of the Falcor Buffer
            int visTexSizeActual = texWidth * texHeight * 4;
            int toSendSize = VIS_TEX_LEN;
            
            // if compress
            if (mCompression)
            {
                char buffer[70];
                sprintf(buffer, "\n\n= Compressing Texture: Original size: %d =========", toSendSize);
                
                // compress from src: toSendData to dst: NetworkPass::visiblityData
                toSendSize = CompressTextureLZ4(visTexSizeActual, toSendData, compressionBuffer.get());
                
                sprintf(buffer, "\n\n= Compressed Texture: Compressed size: %d =========", toSendSize);
                OutputDebugStringA(buffer);

                // now we send the compressed data instead
                toSendData = compressionBuffer.get();
            }

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

/// <summary>
/// Server is set to listen for incoming data from the client until the connection is closed down.
/// Server will take incoming bytes and process it based on the given texture width and height.
/// </summary>
/// <param name="pRenderContext">- render context</param>
/// <param name="pResManager">- resource manager</param>
/// <param name="texWidth">- texture width</param>
/// <param name="texHeight">- texture height</param>
/// <returns></returns>
bool NetworkManager::ListenServer(RenderContext* pRenderContext, std::shared_ptr<ResourceManager> pResManager, int texWidth, int texHeight)
{
    std::unique_lock<std::mutex> lck(NetworkManager::mMutexServerVisTexTcp);
    int posTexSize = texWidth * texHeight * 16;
    int visTexSize = texWidth * texHeight * 4;

    // Receive until the peer shuts down the connection
    do {
        // Receive the camera position from the sender
        OutputDebugString(L"\n\n= NetworkThread - Awaiting camData receiving over network... =========");
        RecvCameraData(NetworkPass::camData, mClientSocket);
        OutputDebugString(L"\n\n= NetworkThread - camData received over network =========");

        NetworkManager::mCamPosReceivedTcp = true;
        NetworkManager::mCvCamPosReceived.notify_all();

        // Allow rendering using the camPos to begin, and wait for visTex to complete rendering
        OutputDebugString(L"\n\n= NetworkThread - Awaiting visTex to finish rendering... =========");
        while (!NetworkManager::mVisTexCompleteTcp)
            NetworkManager::mCvVisTexComplete.wait(lck);

        // We reset it to false so that we need to wait for NetworkPass::executeServerSend to flag it as true
        // before we can continue sending the next frame
        NetworkManager::mVisTexCompleteTcp = false;

        // Send the visBuffer back to the sender
        OutputDebugString(L"\n\n= NetworkThread - VisTex finished rendering. Awaiting visTex sending over network... =========");
        SendTexture(visTexSize, (char*)NetworkPass::pVisibilityDataServer, mClientSocket);
        OutputDebugString(L"\n\n= NetworkThread - visTex sent over network =========");
    } while (true);

    return true;
}
bool NetworkManager::CloseServerConnectionUdp()
{
    closesocket(mServerUdpSock);
    WSACleanup();
    return true;
}

/// <summary>
/// Attempt to close the server connection gracefully.
/// </summary>
/// <returns></returns>
bool NetworkManager::CloseServerConnection()
{
    int iResult = shutdown(mClientSocket, SD_SEND);
    if (iResult == SOCKET_ERROR) {
        std::string errString = std::string("\n\n= CloseServerConnection - shutdown failed with error: ") + std::to_string(WSAGetLastError());
        OutputDebugString(string_2_wstring(errString).c_str());
        closesocket(mClientSocket);
        WSACleanup();
        return false;
    }

    // Cleanup
    closesocket(mClientSocket);
    WSACleanup();

    return true;
}

/// <summary>
/// Sets up client connection and repeatedly attempt to connect to the given server name and port.
/// Client will make port with arbitrary port number.
/// </summary>
/// <param name="serverName">- server name</param>
/// <param name="serverPort">- server port</param>
/// <returns></returns>
bool NetworkManager::SetUpClient(PCSTR serverName, PCSTR serverPort)
{
    mConnectSocket = INVALID_SOCKET;

    WSADATA wsaData;
    struct addrinfo* result = NULL,
        * ptr = NULL,
        hints;
    int iResult;

    // Initialize Winsock
    iResult = WSAStartup(MAKEWORD(2, 2), &wsaData);
    if (iResult != 0) {
        std::string errString = std::string("\n\n= SetUpClient - WSAStartup failed with error: ") + std::to_string(iResult);
        OutputDebugString(string_2_wstring(errString).c_str());
        return 1;
    }

    ZeroMemory(&hints, sizeof(hints));
    hints.ai_family = AF_UNSPEC;
    hints.ai_socktype = SOCK_STREAM;
    hints.ai_protocol = IPPROTO_TCP;

    // Resolve the server address and port
    iResult = getaddrinfo(serverName, serverPort, &hints, &result);
    if (iResult != 0) {
        std::string errString = std::string("\n\n= SetUpClient - getaddrinfo failed with error: ") + std::to_string(iResult);
        OutputDebugString(string_2_wstring(errString).c_str());
        WSACleanup();
        return false;
    }

    // Attempt to connect to an address until one succeeds
    for (ptr = result; ptr != NULL; ptr = ptr->ai_next) {

        // Create a SOCKET for connecting to server
        mConnectSocket = socket(ptr->ai_family, ptr->ai_socktype,
            ptr->ai_protocol);
        if (mConnectSocket == INVALID_SOCKET) {
            std::string errString = std::string("\n\n= SetUpClient - socket failed with error: ") + std::to_string(WSAGetLastError());
            OutputDebugString(string_2_wstring(errString).c_str());
            WSACleanup();
            return false;
        }

        // Connect to server.
        iResult = connect(mConnectSocket, ptr->ai_addr, (int)ptr->ai_addrlen);
        if (iResult == SOCKET_ERROR) {
            closesocket(mConnectSocket);
            mConnectSocket = INVALID_SOCKET;
            continue;
        }
        break;
    }

    freeaddrinfo(result);

    if (mConnectSocket == INVALID_SOCKET) {
        std::string errString = std::string("\n\n= SetUpClient - Unable to connect to server!");
        OutputDebugString(string_2_wstring(errString).c_str());
        WSACleanup();
        return false;
    }

    return true;
}

bool NetworkManager::SetUpClientUdp(PCSTR serverName, PCSTR serverPort)
{
    int slen = sizeof(mSi_otherUdp);
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
    memset((char*)&mSi_otherUdp, 0, sizeof(mSi_otherUdp));
    mSi_otherUdp.sin_family = AF_INET;
    mSi_otherUdp.sin_port = htons((u_short)std::strtoul(serverPort, NULL, 0));
    inet_pton(AF_INET, serverName, &mSi_otherUdp.sin_addr.S_un.S_addr);
    
    //mSi_otherUdp.sin_addr.S_un.S_addr = inet_addr(serverName);
    return true;
}

void NetworkManager::ListenClientUdp(bool isFirstReceive, bool executeForever)
{
    bool firstClientReceive = isFirstReceive;

    // To be used for decompression
    // 
    // We allocate extra space for the packet's header as it will allow
    // RecvTextureUdp to write directly to the buffer without copying
    std::unique_ptr<char[]> compressionBuffer =
        std::make_unique<char[]>(VIS_TEX_LEN + UdpCustomPacket::headerSizeBytes);
    
    while (true)
    {
        std::chrono::time_point startOfFrame = std::chrono::system_clock::now();
        // dont render the first time because visibilityBuffer doesnt exist yet
        // (new ordering sends camera Data after receive visibilityBuffer)
        /*if (mFirstRender) {
            mFirstRender = false;
            return;
        }*/

        // Await server to send back the visibility pass texture
        OutputDebugString(L"\n\n= Awaiting visTex receiving over network... =========");
        int visTexLen = VIS_TEX_LEN;
        FrameData rcvdFrameData = { visTexLen, clientFrameNum, 0 };
        
        char* visWritingBuffer = NetworkPass::visibilityDataForWritingClient;
        char* toRecvData = NetworkManager::mCompression
            ? compressionBuffer.get() + UdpCustomPacket::headerSizeBytes
            : visWritingBuffer;
        int recvStatus;

        if (firstClientReceive)
        {        
            // Need to take a while to wait for the server,
            // so we use a longer time out for the first time
            recvStatus = RecvTextureUdp(rcvdFrameData, toRecvData, mClientUdpSock,
                                        UDP_FIRST_TIMEOUT_MS);
            // Store the time when the first frame was received
            // (server sends timestamps relative to the time when the first frame was fully rendered)
            if (startTime == std::chrono::milliseconds::zero())
            {
                startTime = getCurrentTime();
            }
            firstClientReceive = false;
        }
        else
        {
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
                // std::this_thread::sleep_for(-timeDifference);
            }
        }

        if (recvStatus == 0)
        {
            char frameDataMessage[90];
            sprintf(frameDataMessage, "\n\n= Discarding frame %d (size: %d, time: %d)\n",
                    rcvdFrameData.frameNumber, rcvdFrameData.frameSize, rcvdFrameData.timestamp);
            OutputDebugStringA(frameDataMessage);
            clientFrameNum++;
        }
        else if (recvStatus == 2)
        {
            char frameDataMessage[129];
            sprintf(frameDataMessage, "\n\n= Discarding frame %d (size: %d, time: %d) and frame %d\n",
                    clientFrameNum, rcvdFrameData.frameSize, rcvdFrameData.timestamp, clientFrameNum + 1);
            OutputDebugStringA(frameDataMessage);
            clientFrameNum += 2;
        }
        else // recvStatus == 1 || recvStatus == 3
        {
            OutputDebugString(L"\n\n= visTex received over network =========");
            char frameDataMessage[89];
            sprintf(frameDataMessage, "\nFrameData: Number: %d, Size: %d, Time: %d\n",
                    rcvdFrameData.frameNumber, rcvdFrameData.frameSize, rcvdFrameData.timestamp);
            OutputDebugStringA(frameDataMessage);
                    
            // if compress
            if (NetworkManager::mCompression)
            {
                DecompressTextureLZ4(visTexLen, visWritingBuffer, rcvdFrameData.frameSize, toRecvData);
            }

            // acquire reading buffer mutex to swap buffers
            {
                std::lock_guard readingLock(NetworkManager::mMutexClientVisTexRead);
                char* tempPtr = NetworkPass::visibilityDataForReadingClient;
                NetworkPass::visibilityDataForReadingClient = NetworkPass::visibilityDataForWritingClient;
                NetworkPass::visibilityDataForWritingClient = tempPtr;
                // mutex and lock are released at the end of scope
            }

            std::chrono::time_point endOfFrame = std::chrono::system_clock::now();
            std::chrono::duration<double> diff = endOfFrame - startOfFrame;
            char printFps[102];
            sprintf(printFps, "\n\n= ListenClientUdp - Frame took %.10f s, estimated FPS: %.2f =========", diff.count(), getFps(diff));
            OutputDebugStringA(printFps);

            if (recvStatus == 3)
            {
                
                char discardMessage[40];
                sprintf(discardMessage, "\n\n= Discarding frame %d\n", clientFrameNum + 1);
                OutputDebugStringA(discardMessage);
                clientFrameNum += 2;
            }
            else // recvStatus == 1
            {
                clientFrameNum++;
            }
        }


        if (!executeForever)
        {
            break;
        }
    }
}

void NetworkManager::SendWhenReadyClientUdp(Scene::SharedPtr mpScene)
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

        OutputDebugString(L"\n\n= Awaiting camData sending over network... =========");
        SendCameraDataUdp(cam, mClientUdpSock);
        OutputDebugString(L"\n\n= camData sent over network =========");

        std::chrono::time_point endOfFrame = std::chrono::system_clock::now();
        std::chrono::duration<double> diff = endOfFrame - startOfFrame;
        char printFps[109];
        sprintf(printFps, "\n\n= SendWhenReadyClientUdp - Frame took %.10f s, estimated FPS: %.2f =========", diff.count(), getFps(diff));
        OutputDebugStringA(printFps);
    }   
}

bool NetworkManager::CloseClientConnectionUdp()
{
    closesocket(mClientUdpSock);
    WSACleanup();
    return true;
}

/// <summary>
/// Attempt to close client connection gracefully.
/// </summary>
/// <returns></returns>
bool NetworkManager::CloseClientConnection()
{
    char recvbuf[DEFAULT_BUFLEN];
    int recvbuflen = DEFAULT_BUFLEN;

    // Shutdown the connection
    int iResult = shutdown(mConnectSocket, SD_SEND);
    if (iResult == SOCKET_ERROR) {
        printf("shutdown failed with error: %d\n", WSAGetLastError());
        closesocket(mConnectSocket);
        WSACleanup();
        return false;
    }

    // Receive until the peer closes the connection
    do {
        iResult = recv(mConnectSocket, recvbuf, recvbuflen, 0);
        if (iResult > 0)
            printf("Bytes received: %d\n", iResult);
        else if (iResult == 0)
            printf("Connection closed\n");
        else
            printf("recv failed with error: %d\n", WSAGetLastError());

    } while (iResult > 0);

    // Cleanup
    closesocket(mConnectSocket);
    WSACleanup();

    return true;
}

/// <summary>
/// Compress the given texture.
/// </summary>
/// <param name="inTexSize">- initial texture size</param>
/// <param name="inTexData">- texture to be compressed</param>
/// <param name="compTexSize">- compressed texture size</param>
/// <returns></returns>
char* NetworkManager::CompressTexture(int inTexSize, char* inTexData, int& compTexSize)
{
    int maxCompLen = OUT_LEN(inTexSize);
    lzo_uint compLen;
    lzo1x_1_compress((unsigned char*)inTexData, (lzo_uint)inTexSize, &NetworkManager::compData[0], &compLen, &wrkmem[0]);
    compTexSize = (int)compLen;
    return (char*)&NetworkManager::compData[0];
}

int NetworkManager::CompressTextureLZ4(int inTexSize, char* inTexData, char* compTexData)
{
    // int LZ4_compress_default(const char* src, char* dst, int srcSize, int dstCapacity);
    int compTexSize = LZ4_compress_default(inTexData, compTexData, inTexSize, inTexSize);
    return compTexSize;
}

/// <summary>
/// Decompress given data.
/// </summary>
/// <param name="outTexSize">- final decompressed texture size</param>
/// <param name="outTexData">- decompressed data</param>
/// <param name="compTexSize">- compressed texture size</param>
/// <param name="compTexData">- compressed texture</param>
void NetworkManager::DecompressTexture(int outTexSize, char* outTexData, int compTexSize, char* compTexData)
{
    lzo_uint new_len = outTexSize;
    lzo1x_decompress((unsigned char*)compTexData, compTexSize, (unsigned char*)outTexData, &new_len, NULL);
}

int NetworkManager::DecompressTextureLZ4(int outTexSize, char* outTexData, int compTexSize, char* compTexData)
{
    // int LZ4_decompress_safe (const char* src, char* dst, int compressedSize, int dstCapacity);
    int outputSize = LZ4_decompress_safe(compTexData, outTexData, compTexSize, outTexSize);
    return outputSize;
}

/// <summary>
/// Receive a texture of a given size from the given socket.
/// </summary>
/// <param name="recvTexSize">- size of texture to be received</param>
/// <param name="recvTexData">- variable to be filed with the texture data</param>
/// <param name="socket">- socket to receive from</param>
void NetworkManager::RecvTexture(int recvTexSize, char* recvTexData, SOCKET& socket)
{
    // If no compression occurs, we write directly to the recvTex with the expected texture size,
    // but if we are using compression, we need to receive a compressed texture to an intermediate
    // array and decompress
    char* recvDest = mCompression ? (char*)&NetworkManager::compData[0] : recvTexData;
    int recvSize = recvTexSize;
    if (mCompression)
    {
        RecvInt(recvSize, socket);
    }

    // Receive the texture
    int recvSoFar = 0;
    while (recvSoFar < recvSize)
    {
        int iResult = recv(socket, &recvDest[recvSoFar], DEFAULT_BUFLEN, 0);
        if (iResult > 0)
        {
            recvSoFar += iResult;
        }
    }

    // Decompress the texture if using compression
    if (mCompression)
    {
        recvTexSize = DecompressTextureLZ4(recvTexSize, recvTexData, recvSize, (char*)&NetworkManager::compData[0]);
    }
}

/// <summary>
/// Send texture of the given size.
/// </summary>
/// <param name="sendTexSize">- size of texture to be sent</param>
/// <param name="sendTexData">- texture</param>
/// <param name="socket">- socket to send from</param>
void NetworkManager::SendTexture(int sendTexSize, char* sendTexData, SOCKET& socket)
{
    // If no compression occurs, we directly send the texture with the expected texture size
    char* srcTex = sendTexData;
    int sendSize = sendTexSize;
    
    // But if compression occurs, we perform compression and send the compressed texture size
    // to the other device
    std::string message = std::string("\n\n= Size of texture: ") + std::to_string(sendSize) + std::string(" =========");
    OutputDebugString(string_2_wstring(message).c_str());

    if (mCompression)
    {
        //srcTex = CompressTexture(sendTexSize, sendTexData, sendSize);
        sendSize = CompressTextureLZ4(sendTexSize, sendTexData, (char*)&NetworkManager::compData[0]);
        SendInt(sendSize, socket);

        // Send the compressed texture
        int sentSoFar = 0;
        while (sentSoFar < sendSize)
        {
            bool lastPacket = sentSoFar > sendSize - DEFAULT_BUFLEN;
            int sizeToSend = lastPacket ? (sendSize - sentSoFar) : DEFAULT_BUFLEN;
            int iResult = send(socket, (char*)&NetworkManager::compData[0], sizeToSend, 0);
            if (iResult != SOCKET_ERROR)
            {
                sentSoFar += iResult;
            }
        }
        return;
    }

    // Send the texture
    int sentSoFar = 0;
    while (sentSoFar < sendSize)
    {
        bool lastPacket = sentSoFar > sendSize - DEFAULT_BUFLEN;
        int sizeToSend = lastPacket ? (sendSize - sentSoFar) : DEFAULT_BUFLEN;
        int iResult = send(socket, &srcTex[sentSoFar], sizeToSend, 0);
        if (iResult != SOCKET_ERROR)
        {
            sentSoFar += iResult;
        }
    }
}

int NetworkManager::RecvTextureUdp(FrameData& frameDataOut, char* outRecvTexData, SOCKET& socketUdp, int timeout)
{
    int recvTexSize = frameDataOut.frameSize;
    int expectedFrameNum = frameDataOut.frameNumber;
    // Initialise numberOfPackets to the expected number without compression
    int numberOfPackets = recvTexSize / UdpCustomPacket::maxPacketSize +
        ((recvTexSize % UdpCustomPacket::maxPacketSize > 0) ? 1 : 0);
    // Try receiving an additional packet if we see a new frame
    int newFrameRecvLimit = 1;

    int receivedDataSoFar = 0;
    uint8_t* dataPtr = reinterpret_cast<uint8_t*>(outRecvTexData);
    char oldHeaderBytes[UdpCustomPacket::headerSizeBytes];

    int seqNumOfFirstPacket = serverSeqNum;
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
        if (!RecvUdpCustom(recvPacketStart, recvHeader, dataPointer, socketUdp, timeout))
        {
            char buffer[73];
            sprintf(buffer, "\n\n= RecvTextureUdp: Failed to receive packet %d =========", serverSeqNum);
            OutputDebugStringA(buffer);
            continue;
        }
        else
        {
            // Skip past packets that belong to an older frame
            if (recvHeader.frameNumber < expectedFrameNum)
            {
                char buffer[93];
                sprintf(buffer, "\n\n= RecvTextureUdp: "
                        "Skipping past packet %d for older frame %d",
                        recvHeader.sequenceNumber, recvHeader.frameNumber);
                OutputDebugStringA(buffer);
                continue;
            }
            else if (recvHeader.frameNumber > expectedFrameNum)
            {
                // If we are just missing the last packet, we continue
                // trying to receive one more packet and hope that it
                // belongs to the current frame.
                if (numOfRecvAttempts < newFrameRecvLimit &&
                    relativeSeqNum == numberOfPackets - 1)
                {
                    // If packets belongs to a newer frame, the next frame
                    // and the current frame are both ruined.
                    char buffer[95];
                    sprintf(buffer, "\n\n= RecvTextureUdp: "
                            "Received a packet %d for newer frame %d",
                            recvHeader.sequenceNumber, recvHeader.frameNumber);        
                    OutputDebugStringA(buffer);
                    sprintf(buffer, "\n\n= RecvTextureUdp: "
                            "Trying again to receive the last packet %d for frame %d",
                            serverSeqNum, expectedFrameNum);
                    OutputDebugStringA(buffer);
                    numOfRecvAttempts++;
                    continue;
                }
                else
                {
                    // If packets belongs to a newer frame, the next frame
                    // and the current frame are both ruined.
                    char buffer[113];
                    sprintf(buffer, "\n\n= RecvTextureUdp: "
                            "Frame %d ruined by packet %d for newer frame %d",
                            expectedFrameNum, recvHeader.sequenceNumber, recvHeader.frameNumber);        
                    OutputDebugStringA(buffer);
                    return 2;
                }
            }
            
            // Remember frame data for the first full packet received
            if (relativeSeqNum == 0)
            {
                numberOfPackets = recvHeader.numOfFramePackets;
                frameDataOut.timestamp = recvHeader.timestamp;
                serverSeqNum = recvHeader.sequenceNumber;
                seqNumOfFirstPacket = serverSeqNum;
            }
            else if (recvHeader.sequenceNumber != serverSeqNum)
            {
                char buffer[107];
                sprintf(buffer, "\n\n= RecvTextureUdp: "
                        "Sequence number does not match, expected %d, received %d",
                        serverSeqNum, recvHeader.sequenceNumber);
                OutputDebugStringA(buffer);
                
                // Skip ahead if possible
                int difference = recvHeader.sequenceNumber - serverSeqNum;
                if (difference > 0)
                {
                    char bufferSet[74];
                    sprintf(bufferSet, "\n\n= RecvTextureUdp: "
                            "Setting expected sequence number to %d",
                            recvHeader.sequenceNumber);
                    OutputDebugStringA(bufferSet);
                    serverSeqNum = recvHeader.sequenceNumber;
                }

                // Reject the entire frame and return
                return 0;
            }

            // Replace the original bytes at the header position, now that we have the header data            
            if (relativeSeqNum > 0)
            {
                for (int j = 0; j < UdpCustomPacket::headerSizeBytes; j++)
                {
                    recvPacketStart[j] = oldHeaderBytes[j];
                }
            }

            // Increment for the next packet
            relativeSeqNum++;
            serverSeqNum++;
            receivedDataSoFar += recvHeader.dataSize;
        }
    }
    frameDataOut.frameSize = receivedDataSoFar;
    OutputDebugString(L"\n\n= RecvTextureUdp: Received texture =========");

    if (numOfRecvAttempts > 0)
    {
        // If we already received the packets of the next frame,
        // we cannot retrieve them as we do not do any copying.
        return 3;
    }
    else
    {
        return 1;
    }
}

void NetworkManager::SendTextureUdp(FrameData frameData, char* sendTexData, SOCKET& socketUdp)
{
    // Variable splitSize controls the size of the split packets
    int32_t splitSize = UdpCustomPacket::maxPacketSize;
    int16_t numOfFramePackets = frameData.frameSize / splitSize +
                                ((frameData.frameSize % splitSize > 0) ? 1 : 0);
    
    // Split the frame data and send
    int currentOffset = 0;
    for (int32_t amountLeft = frameData.frameSize; amountLeft > 0; amountLeft -= splitSize)
    {
        int32_t size = std::min(amountLeft, UdpCustomPacket::maxPacketSize);                                  
        UdpCustomPacketHeader texHeader(serverSeqNum, size, frameData.frameNumber,
                                        numOfFramePackets, frameData.timestamp);

        if (!SendUdpCustom(texHeader, &sendTexData[currentOffset], socketUdp))
        {
            char buffer[70];
            sprintf(buffer, "\n\n= SendTextureUdp: Failed to send packet %d =========",
                    texHeader.sequenceNumber);
            OutputDebugStringA(buffer);
            return;
        }

        serverSeqNum++;
        currentOffset += size;
    }

    OutputDebugString(L"\n\n= SendTextureUdp: Sent texture =========");
}

/// <summary>
/// Receive an integer from the socket.
/// </summary>
/// <param name="recvInt">- variable to receive the integer</param>
/// <param name="s">- socket to accept the integer</param>
/// <returns></returns>
bool NetworkManager::RecvInt(int& recvInt, SOCKET& s)
{
    int32_t ret{};
    char* data = (char*)&ret;
    int amtToRecv = sizeof(ret);
    int recvSoFar = 0;
    do
    {
        int iResult = recv(s, &data[recvSoFar], amtToRecv - recvSoFar, 0);
        if (iResult != SOCKET_ERROR) recvSoFar += iResult;
    } while (recvSoFar < amtToRecv);

    recvInt = ntohl(ret);
    return true;
}

/// <summary>
/// Send an integer from the socket.
/// </summary>
/// <param name="toSend">- integer to send</param>
/// <param name="s">- socket to send from</param>
/// <returns></returns>
bool NetworkManager::SendInt(int toSend, SOCKET& s)
{
    int32_t conv = htonl(toSend);
    char* data = (char*)&conv;
    int amtToSend = sizeof(conv);
    int sentSoFar = 0;
    do
    {
        int iResult = send(s, &data[sentSoFar], amtToSend - sentSoFar, 0);
        if (iResult != SOCKET_ERROR) sentSoFar += iResult;
    } while (sentSoFar < amtToSend);
    return true;
}

/// <summary>
/// Receive the camera data. Camera data is given in an array<float3, 3>.
/// </summary>
/// <param name="cameraData">- array to receive the camera data</param>
/// <param name="s">- socket to receive the data from</param>
/// <returns></returns>
bool NetworkManager::RecvCameraData(std::array<float3, 3>& cameraData, SOCKET& s)
{
    char* data = (char*)&cameraData;
    int amtToRecv = sizeof(std::array<float3, 3>);
    int recvSoFar = 0;
    do
    {
        int iResult = recv(s, &data[recvSoFar], amtToRecv - recvSoFar, 0);
        if (iResult != SOCKET_ERROR) recvSoFar += iResult;
    } while (recvSoFar < amtToRecv);
    return true;
}

/// <summary>
/// Send the camera data. Camera data is given in an array<float3, 3>.
/// </summary>
/// <param name="cam">- camera</param>
/// <param name="s">- socket to send</param>
/// <returns></returns>
bool NetworkManager::SendCameraData(Camera::SharedPtr cam, SOCKET& s)
{
    std::array<float3, 3> cameraData = { cam->getPosition(), cam->getUpVector(), cam->getTarget() };

    char* data = (char*)&cameraData;
    int amtToSend = sizeof(cameraData);
    int sentSoFar = 0;
    do
    {
        int iResult = send(s, &data[sentSoFar], amtToSend - sentSoFar, 0);
        if (iResult != SOCKET_ERROR) sentSoFar += iResult;
    } while (sentSoFar < amtToSend);

    return true;
}

bool NetworkManager::RecvCameraDataUdp(
    std::array<float3, 3>& cameraData,
    std::mutex& mutexForCameraData,
    SOCKET& socketUdp,
    bool useLongTimeout)
{
    // Assumes server is receiving cam data from client
    UdpCustomPacketHeader recvHeader;
    bool hasReceived;
    char* packetData;
    char* recvBuffer = new char[DEFAULT_BUFLEN];
    if (useLongTimeout)
    {
        hasReceived = RecvUdpCustom(recvBuffer, recvHeader, packetData, socketUdp,
                                    UDP_FIRST_TIMEOUT_MS);
    }
    else
    {
        hasReceived = RecvUdpCustom(recvBuffer, recvHeader, packetData, socketUdp);
    }

    if (!hasReceived)
    {
        delete[] recvBuffer;
        char bufferSn[75];
        sprintf(bufferSn, "\n\n= RecvCameraDataUdp: Failed to receive %d =========", clientSeqNum);
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
            cameraData = cameraDataCache.back();
            clientSeqNum++;
            return true;
        }
    }
    else
    {
        // Increment sequence number for next communication
        clientSeqNum++;
        {
            assert(recvHeader.dataSize == sizeof(cameraData));
            uint8_t* dataOut = reinterpret_cast<uint8_t*>(&cameraData);
            
            // Copy the data to the pointer
            std::lock_guard<std::mutex> lock(mutexForCameraData);
            for (int i = 0; i < recvHeader.dataSize; i++)
            {
                dataOut[i] = packetData[i];
            }
        }
        delete[] recvBuffer;

        // Populate the cache
        cameraDataCache.push_back(cameraData);
        if (cameraDataCache.size() > maxCamDataCacheSize)
        {
            cameraDataCache.pop_front();
        }
        return true;
    }
}

bool NetworkManager::SendCameraDataUdp(Camera::SharedPtr camera, SOCKET& socketUdp)
{
    std::array<float3, 3> cameraData = { camera->getPosition(), camera->getUpVector(), camera->getTarget() };
    char* data = reinterpret_cast<char*>(&cameraData);
    // Assumes client sending to server
    UdpCustomPacketHeader headerToSend(clientSeqNum, sizeof(cameraData));
    clientSeqNum++;

    bool wasDataSent = true;
    if (!SendUdpCustom(headerToSend, data, socketUdp))
    {
        OutputDebugString(L"\n\n= SendCameraDataUdp: Failed to send =========");
        wasDataSent = false;
    }
    return wasDataSent;
}

bool NetworkManager::RecvUdpCustom(
    char* dataBuffer,
    UdpCustomPacketHeader& outDataHeader,
    char*& outDataPointer,
    SOCKET& socketUdp,
    int timeout,
    bool storeAddress)
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

    // Try to store a reply address
    if (storeAddress)
    {
        // Set up the address for replying
        memset(reinterpret_cast<char*>(&mSi_otherUdp), 0, sizeof(mSi_otherUdp));
        mSi_otherUdp.sin_family = clientAddr.sin_family;
        mSi_otherUdp.sin_port = clientAddr.sin_port;
        mSi_otherUdp.sin_addr.S_un.S_addr = clientAddr.sin_addr.S_un.S_addr;
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

bool NetworkManager::RecvUdpCustomAndCheck(
    char* dataBuffer,
    UdpCustomPacketHeader& outDataHeader,
    char*& outDataPointer,
    SOCKET& socketUdp,
    int expectedSeqNum,
    int timeout,
    bool storeAddress)
{
    if (!RecvUdpCustom(dataBuffer, outDataHeader, outDataPointer, socketUdp, timeout, storeAddress))
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

bool NetworkManager::SendUdpCustom(UdpCustomPacketHeader& dataHeader, char* dataToSend, SOCKET& socketUdp)
{
    std::unique_ptr<char[]> udpToSend = dataHeader.createUdpPacket(dataToSend);

    // Send the data
    char msgBuffer0[65];
    sprintf(msgBuffer0, "\n\n= SendUdpCustom: Sending packet %d... =========", dataHeader.sequenceNumber);
    OutputDebugStringA(msgBuffer0);

    struct sockaddr* toSocket = reinterpret_cast<sockaddr*>(&mSi_otherUdp);
    int socketLen = sizeof(mSi_otherUdp);
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

std::chrono::milliseconds NetworkManager::getComparisonTimestamp()
{
    return getCurrentTime() - startTime;
}
