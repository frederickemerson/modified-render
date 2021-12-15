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
#include "./UdpCustomPacket.h"
#include "./Semaphore.h"
#include "../NetworkPasses/NetworkUtils.h"
#include "../DxrTutorSharedUtils/HrenderConfig.h"

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

using namespace Falcor;

class ResourceManager;

// Stores the metadata for a specific frame
typedef struct FrameData
{
    int frameSize;      // Total size of the frame in bytes
    int frameNumber;    // Number associated to the current frame
    int timestamp;      // Time offset from the start in milliseconds
} FrameData;

class NetworkManager : public std::enable_shared_from_this<NetworkManager> {

public:
    // Used by Server
    SOCKET mListenSocket = INVALID_SOCKET;
    SOCKET mClientSocket = INVALID_SOCKET;
    SOCKET mServerUdpSock = INVALID_SOCKET;
    struct sockaddr_in mServer, mSsi_other;

    // Used by client
    SOCKET mConnectSocket = INVALID_SOCKET;
    SOCKET mClientUdpSock = INVALID_SOCKET;
    int32_t clientFrameNum = 0;

    // Used by both server and client in UDP communication
    int32_t serverSeqNum = 0;
    int32_t clientSeqNum = 0;
    struct sockaddr_in mSi_otherUdp;

    using SharedPtr = std::shared_ptr<NetworkManager>;
    using SharedConstPtr = std::shared_ptr<const NetworkManager>;

    static SharedPtr create() { return SharedPtr(new NetworkManager()); }

    // Used for thread synchronizing
    // 
    // Client's side
    // Synchronise client sending thread with the rendering
    static Semaphore mSpClientCamPosReadyToSend;
    // Protect the client visibility textures with mutexes
    static std::mutex mMutexClientVisTexRead;  // To lock the reading buffer

    // Server's side
    // Synchronise server sending thread with the rendering
    static Semaphore mSpServerVisTexComplete;
    // Check whether the camera position is updated before rendering
    static Semaphore mSpServerCamPosUpdated;
    // Protect the server visibility textures
    static std::mutex mMutexServerVisTexRead;  // For reading from Falcor Buffer
    // Protect the server's camera data
    static std::mutex mMutexServerCamData;

    // Used for compression
    static bool mCompression;
    static std::vector<char> wrkmem;
    static std::vector<unsigned char> compData;

    // A place to store packets that arrive out-of-order
    // Map of sequence number to the received packet
    std::unordered_map<int32_t, UdpCustomPacketHeader> packetCache;

    // A place to store old camera data
    std::deque<std::array<float3, 3>> cameraDataCache;
    // Maximum number of data entries
    int maxCamDataCacheSize = 5;

    // last camera data sent out to server, this helps us render the GBuffer with matching camera data
    // for now we are just manually getting these 3 camera data points specifically for GBuffer needs
    std::atomic<float> cameraUX = 0;
    std::atomic<float> cameraUY = 0;
    std::atomic<float> cameraUZ = 0;
    std::atomic<float> cameraVX = 0;
    std::atomic<float> cameraVY = 0;
    std::atomic<float> cameraVZ = 0;
    std::atomic<float> cameraWX = 0;
    std::atomic<float> cameraWY = 0;
    std::atomic<float> cameraWZ = 0;

    // Use UDP to receive and send texture data
    // 
    // outRecvTexData - The pointer to the location that the texture
    //                  will be written to. A header's worth of space
    //                  needs to be allocated behind this pointer so
    //                  that we can receive the UDP packet directly
    //                  into the pointer given.
    //
    // There are 4 possible return values:
    // 0              - Current frame is to be discarded due to
    //                  packet loss or reordering.
    // 1              - Current frame is received successfully.
    // 2              - Current frame and next frame is to be
    //                  discarded. The next possible frame that
    //                  can be received will be current + 2.
    // 3              - Current frame is okay, but next frame
    //                  has to be discarded.
    int RecvTextureUdp(FrameData& frameDataOut, char* outRecvTexData, SOCKET& socketUdp,
                       int timeout = UDP_LISTENING_TIMEOUT_MS);
    void SendTextureUdp(FrameData frameData, char* sendTexData, SOCKET& socketUdp);
    // Use UDP to receive and send camera data
    bool RecvCameraDataUdp(std::array<float3, 3>& cameraData,
                           std::mutex& mutexForCameraData,
                           SOCKET& socketUdp,
                           bool useLongTimeout);
    bool SendCameraDataUdp(Camera::SharedPtr camera, SOCKET& socketUdp);
    int CompressTextureLZ4(int inTexSize, char* inTexData, char* compTexData);
    int DecompressTextureLZ4(int outTexSize, char* outTexData, int compTexSize, char* compTexData);

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
                       int timeout = UDP_LISTENING_TIMEOUT_MS,
                       bool storeAddress = false);
    
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
                               int timeout = UDP_LISTENING_TIMEOUT_MS,
                               bool storeAddress = false);

    // SendUdpCustom assumes that the packet to send is smaller than
    // the specified maximum size in UdpCustomPacket::maxPacketSize
    bool SendUdpCustom(UdpCustomPacketHeader& dataHeader, char* dataToSend, SOCKET& socketUdp);

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

    // Client 
    bool SetUpClientUdp(PCSTR serverName, PCSTR serverPort);

    // Client's receiving thread
    // isFirstReceive - If true, use a longer timeout
    //                  on the first run of the loop.
    // executeForever - If true, run infinitely.
    void ListenClientUdp(bool isFirstReceive, bool executeForever);

    // Client's sending thread
    void SendWhenReadyClientUdp(Scene::SharedPtr mpScene);

    bool CloseClientConnectionUdp();

private:
    // The time when the client first receives a rendered frame from the server
    std::chrono::milliseconds startTime = std::chrono::milliseconds::zero();

    // A helper function to get the time from startTime
    std::chrono::milliseconds getComparisonTimestamp();
};
