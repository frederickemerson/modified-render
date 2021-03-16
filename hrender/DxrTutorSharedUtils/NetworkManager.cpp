#include "ResourceManager.h"
#include "../NetworkPasses/NetworkPass.h"
#include "NetworkManager.h"

bool NetworkManager::mPosTexReceived = false;
bool NetworkManager::mVisTexComplete = false;
std::mutex NetworkManager::mMtxVisTexComplete;
std::mutex NetworkManager::mMtxPosTexReceived;
std::mutex NetworkManager::mMutex;
std::condition_variable NetworkManager::mCvVisTexComplete;
std::condition_variable NetworkManager::mCvPosTexReceived;

bool NetworkManager::SetUpServer(PCSTR port, int& outTexWidth, int& outTexHeight)
{
    WSADATA wsaData;
    int iResult;

    NetworkManager::ListenSocket = INVALID_SOCKET;
    NetworkManager::ClientSocket = INVALID_SOCKET;
    
    struct addrinfo* result = NULL;
    struct addrinfo hints;

    OutputDebugString(L"\n\n===== Pre-Falcor Init - NetworkManager::SetUpServer - PIPELINE SERVER SETTING UP ================================\n\n");

    // Initialize Winsock
    iResult = WSAStartup(MAKEWORD(2, 2), &wsaData);
    if (iResult != 0) {
        std::string errString = std::string("\n\n= Pre-Falcor Init - getaddrinfo failed with error: ") + std::to_string(iResult);
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
    NetworkManager::ListenSocket = socket(result->ai_family, result->ai_socktype, result->ai_protocol);
    if (NetworkManager::ListenSocket == INVALID_SOCKET) {
        std::string errString = std::string("\n\n= Pre-Falcor Init - socket failed with error: ") + std::to_string(WSAGetLastError());
        OutputDebugString(string_2_wstring(errString).c_str());
        freeaddrinfo(result);
        WSACleanup();
        return false;
    }

    // Setup the TCP listening socket
    iResult = bind(NetworkManager::ListenSocket, result->ai_addr, (int)result->ai_addrlen);
    if (iResult == SOCKET_ERROR) {
        std::string errString = std::string("\n\n= Pre-Falcor Init - bind failed with error: ") + std::to_string(WSAGetLastError());
        OutputDebugString(string_2_wstring(errString).c_str());
        freeaddrinfo(result);
        closesocket(NetworkManager::ListenSocket);
        WSACleanup();
        return false;
    }

    freeaddrinfo(result);

    OutputDebugString(L"\n\n= Pre-Falcor Init - SETUP COMPLETE\n\n");

    // Listening for client socket
    OutputDebugString(L"\n\n= Pre-Falcor Init - Trying to listen for client... =========\n\n");
    iResult = listen(NetworkManager::ListenSocket, SOMAXCONN);
    if (iResult == SOCKET_ERROR)
    {
        std::string errString = std::string("\n\n= Pre-Falcor Init - listen failed with error: ") + std::to_string(WSAGetLastError());
        OutputDebugString(string_2_wstring(errString).c_str());
        closesocket(NetworkManager::ListenSocket);
        WSACleanup();
        return false;
    }
    // Accept the client socket
    OutputDebugString(L"\n\n= Pre-Falcor Init - Trying to accept client... =========\n\n");
    NetworkManager::ClientSocket = accept(NetworkManager::ListenSocket, NULL, NULL);
    if (ClientSocket == INVALID_SOCKET)
    {
        std::string errString = std::string("\n\n= Pre-Falcor Init - accept failed with error: ") + std::to_string(WSAGetLastError());
        OutputDebugString(string_2_wstring(errString).c_str());
        closesocket(NetworkManager::ListenSocket);
        WSACleanup();
        return false;
    }
    // No longer need server socket
    closesocket(NetworkManager::ListenSocket);
    OutputDebugString(L"\n\n= Pre-Falcor Init - Connection with client established =========\n\n");

    // Get the client texture width/height
    OutputDebugString(L"\n\n= Pre-Falcor Init - Getting client texture width/height... =========\n\n");
    RecvInt(outTexWidth);
    RecvInt(outTexHeight);
    OutputDebugString(L"\n\n= Pre-Falcor Init - Texture width/height received =========\n\n");

    return true;
}

bool NetworkManager::ListenServer(RenderContext* pRenderContext, std::shared_ptr<ResourceManager> pResManager)
{
    int iResult;
    std::unique_lock<std::mutex> lck(NetworkManager::mMutex);
        
    // Receive until the peer shuts down the connection
    do {
        // Receive the position texture from the sender
        OutputDebugString(L"\n\n= NetworkThread - Awaiting posTex receiving over network... =========\n\n");
        int recvSoFar = 0;
        while (recvSoFar < POS_TEX_LEN) {
            iResult = recv(NetworkManager::ClientSocket, (char *)&NetworkPass::posData[recvSoFar], DEFAULT_BUFLEN, 0);
            if (iResult > 0) {
                recvSoFar += iResult;
            }
        }
        OutputDebugString(L"\n\n= NetworkThread - Position texture received over network =========\n\n");

        // Allow rendering using the posTex to begin, and wait for visTex to complete rendering
        NetworkManager::mPosTexReceived = true;
        NetworkManager::mCvPosTexReceived.notify_all();
        OutputDebugString(L"\n\n= NetworkThread - Awaiting visTex to finish rendering... =========\n\n");
        while (!NetworkManager::mVisTexComplete)
            NetworkManager::mCvVisTexComplete.wait(lck);
        // We reset it to false so that we need to wait for NetworkPass::executeServerSend to flag it as true
        // before we can continue sending the next frame
        NetworkManager::mVisTexComplete = false;

        // Send the visBuffer back to the sender
        OutputDebugString(L"\n\n= NetworkThread - VisTex finished rendering. Awaiting visTex sending over network... =========\n\n");
        int sentSoFar = 0;
        while (sentSoFar < VIS_TEX_LEN) {
            bool lastPacket = sentSoFar > VIS_TEX_LEN - DEFAULT_BUFLEN;
            int sizeToSend = lastPacket * (VIS_TEX_LEN - sentSoFar) + !lastPacket * DEFAULT_BUFLEN;
            int iResult = send(NetworkManager::ClientSocket, (char*)&NetworkPass::visibilityData[sentSoFar], sizeToSend, 0);
            if (iResult != SOCKET_ERROR) {
                sentSoFar += iResult;
            }
        }
        OutputDebugString(L"\n\n= NetworkThread - visTex sent over network =========\n\n");
        OutputDebugString(L"\n================================Bytes SENT BACK================================\n");

    } while (true);

    return true;
}

bool NetworkManager::CloseServerConnection()
{
    int iResult = shutdown(NetworkManager::ClientSocket, SD_SEND);
    if (iResult == SOCKET_ERROR) {
        std::string errString = std::string("\n\n= CloseServerConnection - shutdown failed with error: ") + std::to_string(WSAGetLastError());
        OutputDebugString(string_2_wstring(errString).c_str());
        closesocket(NetworkManager::ClientSocket);
        WSACleanup();
        return false;
    }

    // Cleanup
    closesocket(NetworkManager::ClientSocket);
    WSACleanup();

    return true;
}

bool NetworkManager::SetUpClient(PCSTR serverName, PCSTR serverPort)
{
    NetworkManager::ConnectSocket = INVALID_SOCKET;
    
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
        NetworkManager::ConnectSocket = socket(ptr->ai_family, ptr->ai_socktype,
            ptr->ai_protocol);
        if (NetworkManager::ConnectSocket == INVALID_SOCKET) {
            std::string errString = std::string("\n\n= SetUpClient - socket failed with error: ") + std::to_string(WSAGetLastError());
            OutputDebugString(string_2_wstring(errString).c_str());
            WSACleanup();
            return false;
        }

        // Connect to server.
        iResult = connect(NetworkManager::ConnectSocket, ptr->ai_addr, (int)ptr->ai_addrlen);
        if (iResult == SOCKET_ERROR) {
            closesocket(NetworkManager::ConnectSocket);
            NetworkManager::ConnectSocket = INVALID_SOCKET;
            continue;
        }
        break;
    }

    freeaddrinfo(result);

    if (NetworkManager::ConnectSocket == INVALID_SOCKET) {
        std::string errString = std::string("\n\n= SetUpClient - Unable to connect to server!");
        OutputDebugString(string_2_wstring(errString).c_str());
        WSACleanup();
        return false;
    }

    return true;
}

bool NetworkManager::SendDataFromClient(const std::vector<uint8_t>& data, int len, int flags, const std::vector<uint8_t>& out_data)
{
    // Send posBuffer until finishes
    int sentSoFar = 0;
    while (sentSoFar < data.size()){//POS_TEX_LEN) {
        bool lastPacket = sentSoFar > POS_TEX_LEN - DEFAULT_BUFLEN;
        int sizeToSend = lastPacket ? (POS_TEX_LEN - sentSoFar) : DEFAULT_BUFLEN;
        int iResult = send(NetworkManager::ConnectSocket, (char*)&data[sentSoFar], sizeToSend, 0);
        if (iResult != SOCKET_ERROR) {
            sentSoFar += iResult;
        }
    }

    OutputDebugString(L"Data is SENT from client");

    // Receive until finish
    int recvSoFar = 0;
    while (recvSoFar < VIS_TEX_LEN) {
        int iRecv = recv(NetworkManager::ConnectSocket, (char *)&out_data[recvSoFar], DEFAULT_BUFLEN, 0);
        if (iRecv != SOCKET_ERROR) {
            recvSoFar += iRecv;
        }
    }

    OutputDebugString(L"Data is RECEIVED from client");

    return true;
}

bool NetworkManager::CloseClientConnection()
{
    char recvbuf[DEFAULT_BUFLEN];
    int recvbuflen = DEFAULT_BUFLEN;

    // Shutdown the connection
    int iResult = shutdown(NetworkManager::ConnectSocket, SD_SEND);
    if (iResult == SOCKET_ERROR) {
        printf("shutdown failed with error: %d\n", WSAGetLastError());
        closesocket(ConnectSocket);
        WSACleanup();
        return false;
    }

    // Receive until the peer closes the connection
    do {
        iResult = recv(NetworkManager::ConnectSocket, recvbuf, recvbuflen, 0);
        if (iResult > 0)
            printf("Bytes received: %d\n", iResult);
        else if (iResult == 0)
            printf("Connection closed\n");
        else
            printf("recv failed with error: %d\n", WSAGetLastError());

    } while (iResult > 0);

    // Cleanup
    closesocket(ConnectSocket);
    WSACleanup();

    return true;
}

bool NetworkManager::RecvInt(int& recvInt)
{
    int32_t ret{};
    char* data = (char*)&ret;
    int amtToRecv = sizeof(ret);
    int recvSoFar = 0;
    do
    {
        int iResult = recv(NetworkManager::ConnectSocket, &data[recvSoFar], amtToRecv - recvSoFar, 0);
        if (iResult != SOCKET_ERROR) recvSoFar += iResult;
    } while (recvSoFar < amtToRecv);

    recvInt = ntohl(ret);
    return true;
}

bool NetworkManager::SendInt(int toSend)
{
    int32_t conv = htonl(toSend);
    char* data = (char*)&conv;
    int amtToSend = sizeof(conv);
    int sentSoFar = 0;
    do
    {
        int iResult = send(NetworkManager::ConnectSocket, &data[sentSoFar], amtToSend - sentSoFar, 0);
        if (iResult != SOCKET_ERROR) sentSoFar += iResult;
    } while (sentSoFar < amtToSend);
    return true;
}


