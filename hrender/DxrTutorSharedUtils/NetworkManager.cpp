#include "NetworkManager.h"

bool NetworkManager::SetUpServer(PCSTR port)
{
    WSADATA wsaData;
    int iResult;

    NetworkManager::ListenSocket = INVALID_SOCKET;
    NetworkManager::ClientSocket = INVALID_SOCKET;
    
    struct addrinfo* result = NULL;
    struct addrinfo hints;

    // Initialize Winsock
    iResult = WSAStartup(MAKEWORD(2, 2), &wsaData);
    if (iResult != 0) {
        printf("WSAStartup failed with error: %d\n", iResult);
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
        printf("getaddrinfo failed with error: %d\n", iResult);
        WSACleanup();
        return false;
    }

    // Create a SOCKET for connecting to server
    NetworkManager::ListenSocket = socket(result->ai_family, result->ai_socktype, result->ai_protocol);
    if (NetworkManager::ListenSocket == INVALID_SOCKET) {
        printf("socket failed with error: %ld\n", WSAGetLastError());
        freeaddrinfo(result);
        WSACleanup();
        return false;
    }

    // Setup the TCP listening socket
    iResult = bind(NetworkManager::ListenSocket, result->ai_addr, (int)result->ai_addrlen);
    if (iResult == SOCKET_ERROR) {
        printf("bind failed with error: %d\n", WSAGetLastError());
        freeaddrinfo(result);
        closesocket(NetworkManager::ListenSocket);
        WSACleanup();
        return false;
    }

    freeaddrinfo(result);

    return false;
}

bool NetworkManager::AcceptAndListenServer()
{
    int iResult;
    char recvbuf[DEFAULT_BUFLEN];
    int recvbuflen = DEFAULT_BUFLEN;
    int iSendResult;

    iResult = listen(NetworkManager::ListenSocket, SOMAXCONN);
    if (iResult == SOCKET_ERROR) {
        printf("listen failed with error: %d\n", WSAGetLastError());
        closesocket(NetworkManager::ListenSocket);
        WSACleanup();
        return false;
    }

    // Accept a client socket
    NetworkManager::ClientSocket = accept(NetworkManager::ListenSocket, NULL, NULL);
    if (ClientSocket == INVALID_SOCKET) {
        printf("accept failed with error: %d\n", WSAGetLastError());
        closesocket(NetworkManager::ListenSocket);
        WSACleanup();
        return false;
    }

    // No longer need server socket
    closesocket(NetworkManager::ListenSocket);

    // Receive until the peer shuts down the connection
    do {

        iResult = recv(NetworkManager::ClientSocket, recvbuf, recvbuflen, 0);
        if (iResult > 0) {
            printf("Bytes received: %d\n", iResult);

            // Echo the buffer back to the sender
            iSendResult = send(NetworkManager::ClientSocket, recvbuf, iResult, 0);
            if (iSendResult == SOCKET_ERROR) {
                printf("send failed with error: %d\n", WSAGetLastError());
                closesocket(NetworkManager::ClientSocket);
                WSACleanup();
                return false;
            }
            printf("Bytes sent: %d\n", iSendResult);
        }
        else if (iResult == 0)
            printf("Connection closing...\n");
        else {
            printf("recv failed with error: %d\n", WSAGetLastError());
            closesocket(NetworkManager::ClientSocket);
            WSACleanup();
            return false;
        }

    } while (iResult > 0);

    return true;
}

bool NetworkManager::CloseServerConnection()
{
    int iResult = shutdown(NetworkManager::ClientSocket, SD_SEND);
    if (iResult == SOCKET_ERROR) {
        printf("shutdown failed with error: %d\n", WSAGetLastError());
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
        printf("WSAStartup failed with error: %d\n", iResult);
        return 1;
    }

    ZeroMemory(&hints, sizeof(hints));
    hints.ai_family = AF_UNSPEC;
    hints.ai_socktype = SOCK_STREAM;
    hints.ai_protocol = IPPROTO_TCP;

    // Resolve the server address and port
    iResult = getaddrinfo(serverName, serverPort, &hints, &result);
    if (iResult != 0) {
        printf("getaddrinfo failed with error: %d\n", iResult);
        WSACleanup();
        return false;
    }

    // Attempt to connect to an address until one succeeds
    for (ptr = result; ptr != NULL; ptr = ptr->ai_next) {

        // Create a SOCKET for connecting to server
        NetworkManager::ConnectSocket = socket(ptr->ai_family, ptr->ai_socktype,
            ptr->ai_protocol);
        if (NetworkManager::ConnectSocket == INVALID_SOCKET) {
            printf("socket failed with error: %ld\n", WSAGetLastError());
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
        printf("Unable to connect to server!\n");
        WSACleanup();
        return false;
    }

    return true;
}

bool NetworkManager::SendDataFromClient(const char* data, int len, int flags)
{
    // Send an initial buffer
    int iResult = send(NetworkManager::ConnectSocket, data, len, flags);
    if (iResult == SOCKET_ERROR) {
        printf("send failed with error: %d\n", WSAGetLastError());
        closesocket(NetworkManager::ConnectSocket);
        WSACleanup();
        return false;
    }

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
