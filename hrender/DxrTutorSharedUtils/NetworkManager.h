/**********************************************************************************************************************
# Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without modification, are permitted provided that the
# following conditions are met:
#  * Redistributions of code must retain the copyright notice, this list of conditions and the following disclaimer.
#  * Neither the name of NVIDIA CORPORATION nor the names of its contributors may be used to endorse or promote products
#    derived from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.  IN NO EVENT
# SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA,
# OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
# LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
# ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
**********************************************************************************************************************/

#pragma once

#define WIN32_LEAN_AND_MEAN

#include "Falcor.h"
#include <windows.h>
#include <winsock2.h>
#include <ws2tcpip.h>
#include <stdlib.h>
#include <stdio.h>
#include "../Libraries/minilzo.h"

// Need to link with Ws2_32.lib, Mswsock.lib, and Advapi32.lib
#pragma comment (lib, "Ws2_32.lib")
#pragma comment (lib, "Mswsock.lib")
#pragma comment (lib, "AdvApi32.lib")

#define DEFAULT_BUFLEN 65536
#define DEFAULT_PORT "27015"
#define POS_TEX_LEN 33177600 // 16 * 1920 * 1080 //32593920
#define VIS_TEX_LEN 8294400 // 4 * 1920 * 1080 //800000 

#define OUT_LEN(in_len) (in_len + in_len / 16 + 64 + 3)

using namespace Falcor;

class ResourceManager;
class NetworkPass;

class NetworkManager : public std::enable_shared_from_this<NetworkManager> {

public:
    // Used by Server
    SOCKET mListenSocket = INVALID_SOCKET;
    SOCKET mClientSocket = INVALID_SOCKET;
    SOCKET mSUdpS;
    struct sockaddr_in mServer, mSsi_other;

    // Used by client
    SOCKET mConnectSocket = INVALID_SOCKET;
    SOCKET mUdpS; 
    struct sockaddr_in mSi_otherUdp;

    using SharedPtr = std::shared_ptr<NetworkManager>;
    using SharedConstPtr = std::shared_ptr<const NetworkManager>;

    static SharedPtr create() { return SharedPtr(new NetworkManager()); }

    // Used for thread synchronizing
    static bool mCamPosReceived;
    static bool mVisTexComplete;
    static std::mutex mMutex;
    static std::condition_variable mCvCamPosReceived;
    static std::condition_variable mCvVisTexComplete;

    // Used for compression
    static bool mCompression;
    static std::vector<char> wrkmem;
    static std::vector<unsigned char> compData;

    // Used to send and receive data over the network
    void RecvTexture(int recvTexSize, char* recvTexData, SOCKET& socket);
    void SendTexture(int visTexSize, char* sendTexData, SOCKET& socket);
    bool RecvInt(int& recvInt, SOCKET& s);
    bool SendInt(int toSend, SOCKET& s);
    bool RecvCameraData(std::array<float3, 3>& cameraData, SOCKET& s);
    bool SendCameraData(Camera::SharedPtr cam, SOCKET& s);
    char* CompressTexture(int inTexSize, char* inTexData, int& compTexSize);
    void DecompressTexture(int outTexSize, char* outTexData, int compTexSize, char* compTexData);

    // Server
    // Set up the sockets and connect to a client, and output the client's texture width/height
    bool SetUpServer(PCSTR port, int& outTexWidth, int& outTexHeight);
    bool SetUpServerUdp(PCSTR port);
    bool ListenServer(RenderContext* pRenderContext, std::shared_ptr<ResourceManager> pResManager, int texWidth, int texHeight);
    bool ListenServerUdp(RenderContext* pRenderContext, std::shared_ptr<ResourceManager> pResManager, int texWidth, int texHeight);
    bool CloseServerConnection();
    bool CloseServerConnectionUdp();

    // Client 
    bool SetUpClient(PCSTR serverName, PCSTR serverPort);
    bool SetUpClientUdp(PCSTR serverName, PCSTR serverPort);
    void RecvTextureUdp(int recvTexSize, char* recvTexData, SOCKET& socket);
    bool CloseClientConnectionUdp();

    bool CloseClientConnection();
};
