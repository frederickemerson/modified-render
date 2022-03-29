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
#include <atomic>
#include <deque>
#include <unordered_map>
#include "../DxrTutorSharedUtils/UdpCustomPacket.h"
#include "../DxrTutorSharedUtils/Semaphore.h"
#include "../NetworkPasses/NetworkUtils.h"
#include "../DxrTutorSharedUtils/ResourceManager.h"
#include "../DxrTutorSharedUtils/RenderConfig.h"
#include "FrameData.h"

// for artificial delay
#include <chrono>
#include <thread>

// Need to link with Ws2_32.lib, Mswsock.lib, and Advapi32.lib
#pragma comment (lib, "Ws2_32.lib")
#pragma comment (lib, "Mswsock.lib")
#pragma comment (lib, "AdvApi32.lib")

#define DEFAULT_BUFLEN 65536
#define DEFAULT_PORT "27015"
#define DEFAULT_PORT_UDP "1505"
#define POS_TEX_LEN 33177600 // 16 * 1920 * 1080 //32593920
#define VIS_TEX_LEN 8294400 // 4 * 1920 * 1080 //800000

// While waiting for the first packet from the client, wait
// this amount of time in milliseconds before giving up
#define UDP_FIRST_TIMEOUT_MS 1000000

// While listening for a specific sequence number with texture data,
// wait this amount of time in milliseconds before giving up
#define UDP_LISTENING_TIMEOUT_MS 100

// Wait a longer time for camera data
#define UDP_CAMERA_DATA_TIMEOUT_MS 20000

#define OUT_LEN(in_len) (in_len + in_len / 16 + 64 + 3)

#define MAX_NUM_CLIENT 4

using namespace Falcor;

class ResourceManager;

class ServerNetworkManager : public std::enable_shared_from_this<ServerNetworkManager> {

public:
    // Used by Server
    SOCKET mServerUdpSock = INVALID_SOCKET;
    struct sockaddr_in mServer;
    std::vector<sockaddr_in> mClientAddresses;

    // Used by both server and client in UDP communication
    std::vector<int32_t> serverSeqNum;
    std::vector<int32_t> clientSeqNum;
    std::vector<int32_t> clientFrameNum;

    using SharedPtr = std::shared_ptr<ServerNetworkManager>;
    using SharedConstPtr = std::shared_ptr<const ServerNetworkManager>;

    static SharedPtr create() { return SharedPtr(new ServerNetworkManager()); }

    // Used for thread synchronizing
    // Synchronise server sending thread with the rendering
    static Semaphore mSpServerVisTexComplete;
    // Check whether the camera position is updated before rendering
    static std::array<Semaphore, MAX_NUM_CLIENT> mClientCamPosUpdated;
    // for all camera changes
    static Semaphore mSpServerCamPosUpdated;
    // Protect the server visibility textures
    static std::mutex mMutexServerVisTexRead;  // For reading from Falcor Buffer

    // A place to store packets that arrive out-of-order
    // Map of sequence number to the received packet
    std::unordered_map<int32_t, UdpCustomPacketHeader> packetCache;

    // A place to store old camera data
    std::deque<std::array<float3, 3>> cameraDataCache;
    // Maximum number of data entries
    int maxCamDataCacheSize = 5;

    // Function for getting input buffers
    std::function<char* ()> mGetInputBuffer;
    std::function<int ()> mGetInputBufferSize; 

    // client queue to send
    std::queue<int> sendClientQueue;
    std::map<ULONG, int> mapClientAddressToIndex;

    void SendTextureUdp(FrameData frameData, char* sendTexData, int clientIndex, SOCKET& socketUdp);
    // Use UDP to receive and send camera data
    bool RecvCameraDataUdp(std::vector<std::array<float3, 3>>& cameraData,
                           std::array<std::mutex, MAX_NUM_CLIENT>& mutexCameraData,
                           SOCKET& socketUdp,
                           bool useLongTimeout);

    // Receive data with UDP custom protocol
    // Returns false if an error was encountered
    //
    // dataBuffer     - Pointer to the buffer that the received data
    //                  will be written to, together with its header. 
    //                  The length in bytes of the buffer should be
    //                  greater than or equal to the maximum size
    //                  of a UDP packet (65507 bytes).
    // outDataHeader  - The reference to the header object which will
    //                  be populated with the correct information.
    // outDataPointer - Another pointer to the data buffer, but this
    //                  will be set to point to the beginning of the
    //                  actual data without the custom packet header.
    // socketUdp      - The socket to use for receiving.
    // timeout        - The timeout to be used for receiving.
    // storeAddress   - Set to true to remember the return address.
    bool RecvUdpCustom(char* dataBuffer,
                       UdpCustomPacketHeader& outDataHeader,
                       char*& outDataPointer,
                       SOCKET& socketUdp,
                        int& fromClientIndex,
                       int timeout = UDP_LISTENING_TIMEOUT_MS);
    
    // Same as RecvUdpCustom, but discards the packet if the
    // sequence number does not match the one that was given.
    //
    // dataBuffer     - Pointer to the buffer that the received data
    //                  will be written to, together with its header. 
    //                  The length in bytes of the buffer should be
    //                  greater than or equal to the maximum size
    //                  of a UDP packet (65507 bytes).
    // outDataHeader  - The reference to the header object which will
    //                  be populated with the correct information.
    // outDataPointer - Another pointer to the data buffer, but this
    //                  will be set to point to the beginning of the
    //                  actual data without the custom packet header.
    // socketUdp      - The socket to use for receiving.
    // expectedSeqNum - The expected sequence number of the packet
    //                  that will be received. 
    // timeout        - The timeout to be used for receiving.
    // storeAddress   - Set to true to remember the return address.
    bool RecvUdpCustomAndCheck(char* dataBuffer,
                               UdpCustomPacketHeader& outDataHeader,
                               char*& outDataPointer,
                               SOCKET& socketUdp,
                               int expectedSeqNum,
                               int timeout = UDP_LISTENING_TIMEOUT_MS);

    // SendUdpCustom assumes that the packet to send is smaller than
    // the specified maximum size in UdpCustomPacket::maxPacketSize
    bool SendUdpCustom(UdpCustomPacketHeader& dataHeader, char* dataToSend, int clientIndex, SOCKET& socketUdp);

    // Server
    // Set up UDP socket and listen for client's texture width/height
    bool SetUpServerUdp(PCSTR port, int& outTexWidth, int& outTexHeight);
    // Server's receiving thread
    // Listen to UDP packets with custom protocol
    bool ListenServerUdp(bool executeForever, bool useLongTimeout);
    // Server's sending thread
    void SendWhenReadyServerUdp(RenderContext* pRenderContext,
                                std::shared_ptr<ResourceManager> pResManager,
                                int texWidth,
                                int texHeight);

    bool CloseServerConnectionUdp();

private:
        bool compression = true;
};
