#include "ResourceManager.h"
#include "../NetworkPasses/NetworkPass.h"
#include "NetworkManager.h"

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
    if ((mServerUdpSock = socket(AF_INET, SOCK_DGRAM, 0)) == INVALID_SOCKET)
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
    if (!RecvUdpCustom(firstPacket, mServerUdpSock, UDP_FIRST_TIMEOUT_MS))
    {
        OutputDebugString(L"\n\n= Pre-Falcor Init - FAILED to receive UDP packet from client =========");
        return false;
    }
    
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

void NetworkManager::RecvTextureUdp(int recvTexSize, char* recvTexData, SOCKET& socketUdp)
{
    //char message[DEFAULT_BUFLEN];
    //char buf[DEFAULT_BUFLEN];
    int slen = sizeof(mSi_otherUdp);
    ////start communication
    //while (1)
    //{
    //    printf("Enter message : ");
    //    std::cin >> message;


    //    //send the message
    //    if (sendto(mClientUdpSock, message, (int)strlen(message), 0, (struct sockaddr*) & mSi_otherUdp, slen) == SOCKET_ERROR)
    //    {
    //        printf("sendto() failed with error code : %d", WSAGetLastError());
    //        exit(EXIT_FAILURE);
    //    }

    //    //receive a reply and print it
    //    //clear the buffer by filling null, it might have previously received data
    //    memset(buf, '\0', DEFAULT_BUFLEN);
    //    //try to receive some data, this is a blocking call
    //    if (recvfrom(mClientUdpSock, buf, DEFAULT_BUFLEN, 0, (struct sockaddr*) & mSi_otherUdp, &slen) == SOCKET_ERROR)
    //    {
    //        printf("recvfrom() failed with error code : %d", WSAGetLastError());
    //        exit(EXIT_FAILURE);
    //    }

    //    puts(buf);
    //}
    // If no compression occurs, we write directly to the recvTex with the expected texture size,
    // but if we are using compression, we need to receive a compressed texture to an intermediate
    // array and decompress

    // Server is going to send texture via UDP so it waits for the client to send an int via
    // TCP to synchronize.
    int dummy = 102342;
    // SendInt(dummy, socketTcp);
    // RecvInt(dummy, socketTcp); // Block until server sends
    char message[] = "this is the message from client";
    sendto(socketUdp, message, (int)strlen(message), 0, (struct sockaddr*)&mSi_otherUdp, slen);

    char* recvDest = mCompression ? (char*)&NetworkManager::compData[0] : recvTexData;
    int recvSize = recvTexSize;
    if (mCompression)
    {
        OutputDebugString(L"\n\n= RecvTexture: Receiving int... =========");
        // RecvInt(recvSize, socketTcp);
        OutputDebugString(L"\n\n= RecvTexture: received int =========");

    }

    // Receive the texture
    int recvSoFar = 0;
    OutputDebugString(L"\n\n= RecvTexture: Receiving tex... =========");

    while (recvSoFar < recvSize)
    {
        int iResult = recvfrom(socketUdp, &recvDest[recvSoFar], DEFAULT_BUFLEN, 0, (struct sockaddr*) & mSi_otherUdp, &slen);

        //int iResult = recv(socketTcp, &recvDest[recvSoFar], DEFAULT_BUFLEN, 0);
        if (iResult > 0)
        {
            recvSoFar += iResult;
        }
    }
    OutputDebugString(L"\n\n= RecvTexture: received tex =========");


    // Decompress the texture if using compression

    if (mCompression)
    {
        OutputDebugString(L"\n\n= RecvTexture: Decompressing tex... =========");

        DecompressTexture(recvTexSize, recvTexData, recvSize, (char*)&NetworkManager::compData[0]);
        OutputDebugString(L"\n\n= RecvTexture: Decompressed tex =========");

    }
}

void NetworkManager::SendTextureUdp(int sendTexSize, char* sendTexData, SOCKET& socketUdp)
{
    //char buf[DEFAULT_BUFLEN];    
    //int slen, recv_len;

    int slen = sizeof(mSsi_other);
    //keep listening for data
    //while (1)
    //{
    //    printf("Waiting for data...");
    //    fflush(stdout);

    //    //clear the buffer by filling null, it might have previously received data
    //    memset(buf, '\0', DEFAULT_BUFLEN);

    //    //try to receive some data, this is a blocking call
    //    if ((recv_len = recvfrom(mServerUdpSock, buf, DEFAULT_BUFLEN, 0, (struct sockaddr*) & mSsi_other, &slen)) == SOCKET_ERROR)
    //    {
    //        printf("recvfrom() failed with error code : %d", WSAGetLastError());
    //        exit(EXIT_FAILURE);
    //    }

    //    //print details of the client/peer and the data received
    //    printf("Received packet from\n");
    //    printf("Data: %s\n", buf);

    //    //now reply the client with the same data
    //    if (sendto(mServerUdpSock, buf, recv_len, 0, (struct sockaddr*) & mSsi_other, slen) == SOCKET_ERROR)
    //    {
    //        printf("sendto() failed with error code : %d", WSAGetLastError());
    //        exit(EXIT_FAILURE);
    //    }
    //}

    // Server is going to send texture via UDP, so receive a int via TCP first to signal that
    // client is waiting
    int dummy = 0;
    // RecvInt(dummy, socketTcp);
    // SendInt(dummy, socketTcp);
    std::string msg = std::string("\n\n= Received dummy int: ") + std::to_string(dummy) + std::string(" =========");
    OutputDebugString(string_2_wstring(msg).c_str());
    char buf[DEFAULT_BUFLEN]; memset(buf, '\0', DEFAULT_BUFLEN);
    recvfrom(socketUdp, buf, DEFAULT_BUFLEN, 0, (struct sockaddr*) & mSsi_other, &slen);
    OutputDebugString(string_2_wstring(std::string(buf)).c_str());

    // If no compression occurs, we directly send the texture with the expected texture size
    char* srcTex = sendTexData;
    int sendSize = sendTexSize;

    // But if compression occurs, we perform compression and send the compressed texture size
    // to the other device
    std::string message = std::string("\n\n= Size of texture: ") + std::to_string(sendSize) + std::string(" =========");
    OutputDebugString(string_2_wstring(message).c_str());

    if (mCompression)
    {
        OutputDebugString(L"\n\n= SendTexture: Compressing tex... =========");
        srcTex = CompressTexture(sendTexSize, sendTexData, sendSize);
        OutputDebugString(L"\n\n= SendTexture: Compressed  tex... =========");
        OutputDebugString(L"\n\n= SendTexture: Sending int... =========");
        // SendInt(sendSize, socketTcp);
        OutputDebugString(L"\n\n= SendTexture: Sent int... =========");

    }

    // Send the texture
    OutputDebugString(L"\n\n= SendTexture: Sending tex... =========");
    int sentSoFar = 0;
    while (sentSoFar < sendSize)
    {
        bool lastPacket = sentSoFar > sendSize - DEFAULT_BUFLEN;
        int sizeToSend = lastPacket ? (sendSize - sentSoFar) : DEFAULT_BUFLEN;
        //int iResult = send(socketTcp, &srcTex[sentSoFar], sizeToSend, 0);

        int iResult = sendto(socketUdp, &srcTex[sentSoFar], sizeToSend, 0, (struct sockaddr*) & mSsi_other, slen);

        if (iResult != SOCKET_ERROR)
        {
            sentSoFar += iResult;
        }
        /*std::string mmsg = std::string("<") + std::to_string(sentSoFar) + std::string(">");
        OutputDebugString(string_2_wstring(mmsg).c_str());*/

    }
    OutputDebugString(L"\n\n= SendTexture: Sent tex =========");

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
        DecompressTexture(recvTexSize, recvTexData, recvSize, (char*)&NetworkManager::compData[0]);
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
        srcTex = CompressTexture(sendTexSize, sendTexData, sendSize);
        SendInt(sendSize, socket);
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

bool NetworkManager::RecvUdpCustom(UdpCustomPacket& recvData, SOCKET& socketUdp, int timeout)
{
    int headerSize = UdpCustomPacket::headerSizeBytes;
    char headerData[UdpCustomPacket::headerSizeBytes];
    int headerRecvSoFar = 0;

    // Set timeout for the socket
    if (setsockopt(socketUdp, SOL_SOCKET, SO_RCVTIMEO, reinterpret_cast<char*>(&timeout), sizeof(int)) != 0) {
        char buffer[61];
        sprintf(buffer, "Set socket options failed with error code: %d", WSAGetLastError());
        OutputDebugStringA(buffer);
        return false;
    }

    // Read header for packet size
    do
    {
        sockaddr clientAddr;
        int addrLen;
        int iResult = recvfrom(socketUdp, &headerData[headerRecvSoFar], headerSize - headerRecvSoFar, 0,
                               &clientAddr, &addrLen);
        if (iResult != SOCKET_ERROR)
        {
            headerRecvSoFar += iResult;
        }
    } while (headerRecvSoFar < headerSize);

    int* headerNumbers = reinterpret_cast<int*>(&headerData);
    int seqNum = headerNumbers[0];
    int pktSize = headerNumbers[1];

    // Check the sequence number
    if (seqNum != recvData.sequenceNumber) {
        char buffer[87];
        sprintf(buffer, "Sequence number does not match, expected %d, received %d",
                        recvData.sequenceNumber, seqNum);
        OutputDebugStringA(buffer);
        return false;
    }

    // Receive the rest of the packet
    int dataRecvSoFar = 0;
    do
    {
        int iResult = recv(socketUdp, recvData.getUdpDataPointer(), pktSize - dataRecvSoFar, 0);
        if (iResult != SOCKET_ERROR)
        {
            headerRecvSoFar += iResult;
        }
    } while (dataRecvSoFar < pktSize);

    recvData.packetSize = pktSize;
    return true;
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
            OutputDebugString(L"\n\n= SendUdpCustom: Socket error =========");
            return false;
        }
    }
    
    OutputDebugString(L"\n\n= SendUdpCustom: Sent packets =========");
    return true;
}
