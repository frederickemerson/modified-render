#include "ResourceManager.h"
#include "../NetworkPasses/NetworkPass.h"
#include "NetworkManager.h"

// for LZ4 compression
#include "lz4.h"

bool NetworkManager::mCamPosReceived = false;
bool NetworkManager::mVisTexComplete = false;
bool NetworkManager::mCompression = true;

std::mutex NetworkManager::mMutex;
std::condition_variable NetworkManager::mCvVisTexComplete;
std::condition_variable NetworkManager::mCvCamPosReceived;
std::vector<char> NetworkManager::wrkmem(LZO1X_1_MEM_COMPRESS, 0);
std::vector<unsigned char> NetworkManager::compData(OUT_LEN(POS_TEX_LEN), 0);

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

    // Get the client texture width/height
    UdpCustomPacket firstPacket(0); // expected sequence number of 0
    if (!RecvUdpCustom(firstPacket, mServerUdpSock, UDP_FIRST_TIMEOUT_MS, true))
    {
        OutputDebugString(L"\n\n= Pre-Falcor Init - FAILED to receive UDP packet from client =========");
        return false;
    }
    // Next sequence number should be 1
    currentSeqNum = 1;
    
    // Packet should consist of two ints
    if (firstPacket.packetSize != 8)
    {
        OutputDebugString(L"\n\n= Pre-Falcor Init - FAILED: UDP packet from client has wrong size =========");
        return false;
    }

    int* widthAndHeight = reinterpret_cast<int*>(firstPacket.udpData);
    outTexWidth = widthAndHeight[0];
    outTexHeight = widthAndHeight[1];

    OutputDebugString(L"\n\n= Pre-Falcor Init - Texture width/height received =========");
    char printWidthHeight[52];
    sprintf(printWidthHeight, "\nWidth: %d\nHeight: %d", outTexWidth, outTexHeight);
    OutputDebugStringA(printWidthHeight);
    return true;
}

bool NetworkManager::ListenServerUdp(RenderContext* pRenderContext, std::shared_ptr<ResourceManager> pResManager, int texWidth, int texHeight)
{
    std::unique_lock<std::mutex> lck(NetworkManager::mMutex);
    int posTexSize = texWidth * texHeight * 16;
    int visTexSize = texWidth * texHeight * 4;
    int numFramesRendered = 0;

    // Receive until the peer shuts down the connection
    do
    {
        std::string frameMsg = std::string("\n\n================================ Frame ") + std::to_string(++numFramesRendered) + std::string(" ================================");
        OutputDebugString(string_2_wstring(frameMsg).c_str());

        // Receive the camera position from the sender
        OutputDebugString(L"\n\n= NetworkThread - Awaiting camData receiving over network... =========");
        RecvCameraDataUdp(NetworkPass::camData, mServerUdpSock);
        OutputDebugString(L"\n\n= NetworkThread - camData received over network =========");

        NetworkManager::mCamPosReceived = true;
        NetworkManager::mCvCamPosReceived.notify_all();

        // Allow rendering using the camPos to begin, and wait for visTex to complete rendering
        OutputDebugString(L"\n\n= NetworkThread - Awaiting visTex to finish rendering... =========");
        while (!NetworkManager::mVisTexComplete)
            NetworkManager::mCvVisTexComplete.wait(lck);

        // We reset it to false so that we need to wait for NetworkPass::executeServerSend to flag it as true
        // before we can continue sending the next frame
        NetworkManager::mVisTexComplete = false;

        // Send the visBuffer back to the sender
        OutputDebugString(L"\n\n= NetworkThread - VisTex finished rendering. Awaiting visTex sending over network... =========");
        SendTextureUdp(visTexSize, (char*)&NetworkPass::visibilityData[0], mServerUdpSock);
        OutputDebugString(L"\n\n= NetworkThread - visTex sent over network =========");

        std::string endMsg = std::string("\n\n================================ Frame ") + std::to_string(numFramesRendered) + std::string(" COMPLETE ================================");
        OutputDebugString(string_2_wstring(endMsg).c_str());
    } while (true);

    return true;
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
    std::unique_lock<std::mutex> lck(NetworkManager::mMutex);
    int posTexSize = texWidth * texHeight * 16;
    int visTexSize = texWidth * texHeight * 4;

    // Receive until the peer shuts down the connection
    do {
        // Receive the camera position from the sender
        OutputDebugString(L"\n\n= NetworkThread - Awaiting camData receiving over network... =========");
        RecvCameraData(NetworkPass::camData, mClientSocket);
        OutputDebugString(L"\n\n= NetworkThread - camData received over network =========");

        NetworkManager::mCamPosReceived = true;
        NetworkManager::mCvCamPosReceived.notify_all();

        // Allow rendering using the camPos to begin, and wait for visTex to complete rendering
        OutputDebugString(L"\n\n= NetworkThread - Awaiting visTex to finish rendering... =========");
        while (!NetworkManager::mVisTexComplete)
            NetworkManager::mCvVisTexComplete.wait(lck);

        // We reset it to false so that we need to wait for NetworkPass::executeServerSend to flag it as true
        // before we can continue sending the next frame
        NetworkManager::mVisTexComplete = false;

        // Send the visBuffer back to the sender
        OutputDebugString(L"\n\n= NetworkThread - VisTex finished rendering. Awaiting visTex sending over network... =========");
        SendTexture(visTexSize, (char*)&NetworkPass::visibilityData[0], mClientSocket);
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

void NetworkManager::RecvTextureUdp(int recvTexSize, char* recvTexDataOut, SOCKET& socketUdp, int timeout)
{
    char buffer[73];
    sprintf(buffer, "\n\n= RecvTexSizeUdp: %d =========", recvTexSize);
    OutputDebugStringA(buffer);
    int numberOfPackets = recvTexSize / UdpCustomPacket::maxPacketSize +
                          ((recvTexSize % UdpCustomPacket::maxPacketSize > 0) ? 1 : 0);

    int receivedDataSoFar = 0;
    // If no compression occurs, we write directly to the recvTex with the expected texture size,
    // but if we are using compression, we need to receive a compressed texture to an intermediate
    // array and decompress

    uint8_t* dataPtr = mCompression ? reinterpret_cast<uint8_t*>(&NetworkManager::compData[0]) : reinterpret_cast<uint8_t*>(recvTexDataOut);
    for (int i = 0; i < numberOfPackets; i++)
    {
        // Total offset of the pointer from the start for this packet
        int offset = UdpCustomPacket::maxPacketSize * i;
        UdpCustomPacket toReceive(currentSeqNum);
        if (!RecvUdpCustom(toReceive, socketUdp, timeout))
        {
            /*
            char buffer[73];
            sprintf(buffer, "\n\n= RecvTextureUdp: Failed to receive packet %d =========", currentSeqNum);
            OutputDebugStringA(buffer);
            // Fill missing bits with data from latest
            if (i == numberOfPackets - 1)
            {
                // Last packet, fill in all the other bits
                int dataLeft = recvTexSize - receivedDataSoFar;
                for (int j = 0; j < dataLeft; j++)
                {
                    dataPtr[j] = latestTextureData[receivedDataSoFar + j];
                }
                receivedDataSoFar = recvTexSize;
            }
            else
            {
                // Not the last packet, fill in the maximum amount of bits
                for (int j = 0; j < UdpCustomPacket::maxPacketSize; j++)
                {
                    *dataPtr = latestTextureData[offset + j];
                    dataPtr++;
                }
                receivedDataSoFar += UdpCustomPacket::maxPacketSize;
            }
            // Try to receive the next packet
            currentSeqNum++;
            */
            recvTexSize = receivedDataSoFar;
            char buffer[73];
            sprintf(buffer, "\n\n= Terminated Recv early: Received bytes %d =========", recvTexSize);
            OutputDebugStringA(buffer);
        }
        else
        {
            currentSeqNum++;
            // Copy the packet data to the char* given
            toReceive.copyInto(dataPtr);
            dataPtr += toReceive.packetSize;
            receivedDataSoFar += toReceive.packetSize;
            // Copy the packet data into the latest data cache
            uint8_t* offsetPtr = reinterpret_cast<uint8_t*>(&(latestTextureData[offset]));
            toReceive.copyInto(offsetPtr);
        }
    }

    // Decompress the texture from NetworkManager::compData[0] to recvTexDataOut if using compression
// todo: replace first argument of decompress by the size of original texture sent
    if (mCompression)
    {
        char buffer[70];
        sprintf(buffer, "\n\n= Decompressing Texture: Original size: %d =========", recvTexSize);
        OutputDebugStringA(buffer);
        recvTexSize = DecompressTextureLZ4(VIS_TEX_LEN, recvTexDataOut, recvTexSize, (char*)&NetworkManager::compData[0]);
        sprintf(buffer, "\n\n= Compressed Texture: Uncompressed size: %d =========", recvTexSize);
        OutputDebugStringA(buffer);
    }

    if (receivedDataSoFar != recvTexSize)
    {
        char buffer[137];
        sprintf(buffer, "\n\n= RecvTextureUdp: Error, received size does not match expected size."
                "\nExpected %d, received %d =========", recvTexSize, receivedDataSoFar - 1);
        OutputDebugStringA(buffer);
        return;
    }

    OutputDebugString(L"\n\n= RecvTextureUdp: Received texture =========");
}

void NetworkManager::SendTextureUdp(int sendTexSize, char* sendTexData, SOCKET& socketUdp)
{
    if (mCompression)
    {
        char buffer[70];
        sprintf(buffer, "\n\n= Compressing Texture: Original size: %d =========", sendTexSize);
        sendTexSize = CompressTextureLZ4(sendTexSize, sendTexData, (char*)&NetworkManager::compData[0]);
        sprintf(buffer, "\n\n= Compressed Texture: Compressed size: %d =========", sendTexSize);
        OutputDebugStringA(buffer);
    }

    uint8_t* data = mCompression ? reinterpret_cast<uint8_t*>(&NetworkManager::compData[0]) : reinterpret_cast<uint8_t*>(sendTexData);
    UdpCustomPacket allDataToSend(currentSeqNum, sendTexSize, data);
    std::pair<int32_t, std::vector<UdpCustomPacket>> packets = allDataToSend.splitPacket();
    currentSeqNum = packets.first;
    allDataToSend.releaseDataPointer();

    for (UdpCustomPacket& toSend : packets.second)
    { 
        if (!SendUdpCustom(toSend, socketUdp))
        {
            char buffer[70];
            sprintf(buffer, "\n\n= SendTextureUdp: Failed to send packet %d =========", toSend.sequenceNumber);
            OutputDebugStringA(buffer);
            return;
        }
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

bool NetworkManager::RecvCameraDataUdp(std::array<float3, 3>& cameraData, SOCKET& socketUdp)
{
    UdpCustomPacket toReceive(currentSeqNum);
    if (!RecvUdpCustom(toReceive, socketUdp, UDP_CAMERA_DATA_TIMEOUT_MS))
    {
        OutputDebugString(L"\n\n= RecvCameraDataUdp: Failed to receive =========");
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
            currentSeqNum++;
            return true;
        }
    }
    else
    {
        // Increment sequence number for next communication
        currentSeqNum++;
        // Copy the data to the pointer
        assert(toReceive.packetSize == sizeof(cameraData));
        uint8_t* data = reinterpret_cast<uint8_t*>(&cameraData);
        toReceive.copyInto(data);
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
    uint8_t* data = reinterpret_cast<uint8_t*>(&cameraData);
    UdpCustomPacket toSend(currentSeqNum, sizeof(cameraData), data);
    currentSeqNum++;

    bool wasDataSent = true;
    if (!SendUdpCustom(toSend, socketUdp))
    {
        OutputDebugString(L"\n\n= SendCameraDataUdp: Failed to send =========");
        wasDataSent = false;
    }
    toSend.releaseDataPointer();
    return wasDataSent;
}

bool NetworkManager::RecvUdpCustom(UdpCustomPacket& recvData, SOCKET& socketUdp, int timeout, bool storeAddress)
{
    // Check cache to see if the packet has been received already
    std::unordered_map<int32_t, UdpCustomPacket>::iterator cached = packetCache.find(recvData.sequenceNumber);
    if (cached != packetCache.end())
    {
        char buffer[57];
        sprintf(buffer, "\nRecvUdpCustom: Packet #%d found in cache", recvData.sequenceNumber);
        OutputDebugStringA(buffer);

        if (recvData.udpData == nullptr)
        {
            uint8_t* uintPointer = new uint8_t[cached->second.packetSize];
            recvData.setDataPointer(uintPointer);
        }
        cached->second.copyInto(recvData.udpData);
        cached->second.releaseDataPointer();
        recvData.packetSize = cached->second.packetSize;
        packetCache.erase(cached);
        return true;
    }

    // Number of tries to receive the packet header before failing
    int numberOfTriesForHeader = 1;

    int headerSize = UdpCustomPacket::headerSizeBytes;
    char udpReceiveBuffer[DEFAULT_BUFLEN];
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
    int headerTriesLeft = numberOfTriesForHeader;
    // Read header for packet size
    do
    {
        int iResult = recvfrom(socketUdp, &(udpReceiveBuffer[dataReceivedSoFar]),
                               DEFAULT_BUFLEN, 0, clientAddrPtr, &addrLen);
        if (iResult != SOCKET_ERROR)
        {
            dataReceivedSoFar += iResult;
        }
        else
        {
            char buffer[58];
            sprintf(buffer, "\nRecvUdpCustom: Error receiving header: %d", WSAGetLastError());
            OutputDebugStringA(buffer);
            headerTriesLeft--;
        }
    } while (dataReceivedSoFar < headerSize && headerTriesLeft > 0);

    if (headerTriesLeft == 0)
    {
        char buffer[89];
        sprintf(buffer, "\nTried %d times to receive header for packet %d, giving up",
                        numberOfTriesForHeader, recvData.sequenceNumber);
        OutputDebugStringA(buffer);
        return false;
    }

    int* headerData = reinterpret_cast<int*>(&udpReceiveBuffer);
    int seqNum = headerData[0];
    int dataSize = headerData[1];
    int totalSize = dataSize + headerSize;

    bool doStoreInCache = false;
    // Check the sequence number
    if (seqNum != recvData.sequenceNumber)
    {
        char buffer[88];
        sprintf(buffer, "\nSequence number does not match, expected %d, received %d",
                        recvData.sequenceNumber, seqNum);
        OutputDebugStringA(buffer);
        if (seqNum < recvData.sequenceNumber)
        {
            // Packet received is an older one, regard it as lost
            return false;
        }
        else
        {
            doStoreInCache = true;
            packetCache.emplace(seqNum, UdpCustomPacket(seqNum, dataSize, new uint8_t[dataSize]));
        }
    }

    // Try to store a reply address
    if (storeAddress)
    {
        // Set up the address for replying
        memset(reinterpret_cast<char*>(&mSi_otherUdp), 0, sizeof(mSi_otherUdp));
        mSi_otherUdp.sin_family = clientAddr.sin_family;
        mSi_otherUdp.sin_port = clientAddr.sin_port;
        mSi_otherUdp.sin_addr.S_un.S_addr = clientAddr.sin_addr.S_un.S_addr;
    }

    // Receive the rest of the packet, if needed
    if (dataReceivedSoFar < dataSize)
    {
        do
        {
            int iResult = recvfrom(socketUdp, &(udpReceiveBuffer[dataReceivedSoFar]),
                                   DEFAULT_BUFLEN, 0, clientAddrPtr, &addrLen);
            if (iResult != SOCKET_ERROR)
            {
                dataReceivedSoFar += iResult;
            }
            else
            {
                char buffer[56];
                sprintf(buffer, "\nRecvUdpCustom: Error receiving data: %d", WSAGetLastError());
                OutputDebugStringA(buffer);
            }
        } while (dataReceivedSoFar < dataSize);
    }
    recvData.packetSize = dataSize;

    char* dataPointer;
    if (doStoreInCache)
    {
        dataPointer = packetCache.at(seqNum).getUdpDataPointer();
    }
    else
    {
        dataPointer = recvData.getUdpDataPointer();
    }

    if (dataPointer == nullptr)
    {
        uint8_t* uintPointer = new uint8_t[dataSize];
        recvData.setDataPointer(uintPointer);
        dataPointer = recvData.getUdpDataPointer();
    }

    // Copy data from buffer into UdpCustomPacket object
    for (int i = 0; i < dataSize; i++)
    {
        dataPointer[i] = udpReceiveBuffer[i + headerSize];
    }
    // Return false if packet was stored in cache
    return !doStoreInCache;
}

bool NetworkManager::SendUdpCustom(UdpCustomPacket& dataToSend, SOCKET& socketUdp)
{
    std::unique_ptr<char[]> udpToSend = dataToSend.createUdpPacket();

    // Send the data
    OutputDebugString(L"\n\n= SendUdpCustom: Sending packets... =========");
    struct sockaddr* toSocket = reinterpret_cast<sockaddr*>(&mSi_otherUdp);
    int socketLen = sizeof(mSi_otherUdp);
    int sendSize = UdpCustomPacket::headerSizeBytes + dataToSend.packetSize;
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
            char buffer1[61];
            sprintf(buffer1, "\n\n= SendUdpCustom: Socket error, %d =========", WSAGetLastError());
            OutputDebugStringA(buffer1);

            char buffer2[89];
            sprintf(buffer2, "\n= SendUdpCustom: Failed to send packet with sequence number %d =========",
                    dataToSend.sequenceNumber);
            OutputDebugStringA(buffer2);

            return false;
        }
    }
    
    OutputDebugString(L"\n\n= SendUdpCustom: Sent packets =========");
    return true;
}
