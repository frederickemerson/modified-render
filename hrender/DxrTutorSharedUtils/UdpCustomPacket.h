// A class to encapsulate a custom protocol built on top of UDP

#ifndef UDP_CUSTOM_PACKET_H
#define UDP_CUSTOM_PACKET_H

#include <memory>
#include <vector>

#define UDP_MAX_DATA_LENGTH 65507 // = 65,535 − 8 (UDP header) − 20 (IP header)

class UdpCustomPacketHeader {
public:
    // ======================= HEADER FIELDS =======================
    // 1) Sequence number of this packet
    int32_t sequenceNumber = -1;
    // 2) Frame number which this packet belongs to
    int32_t frameNumber = -1;
    // 3) Packet size in bytes, excluding header size
    uint16_t dataSize = 0;
    // 4) Number of packets that belong to the same frame
    uint16_t numOfFramePackets = 0;
    // 5) Timestamp for when the frame should be played
    int32_t timestamp = -1;


    // ========================== METHODS ==========================
    // Initialise a packet with without frame information
    UdpCustomPacketHeader(int32_t seqNum, uint16_t dtSize);
    // Initialise a packet with all the fields
    UdpCustomPacketHeader(int32_t seqNum, uint16_t dtSize, int32_t frmNum,
                          uint16_t numFrmPkts, int32_t tmStmp);
    UdpCustomPacketHeader() = default;

    UdpCustomPacketHeader(const UdpCustomPacketHeader&) = default;
    UdpCustomPacketHeader& operator=(const UdpCustomPacketHeader&) = default;
    UdpCustomPacketHeader(UdpCustomPacketHeader&&) = default;
    UdpCustomPacketHeader& operator=(UdpCustomPacketHeader&&) = default;
    ~UdpCustomPacketHeader() = default;

    // Creates a UDP packet that can be sent by adding the
    // header fields to the data pointer.
    // 
    // data - Pointer to the buffer containing the data to be sent.
    //        The length in bytes of the buffer pointed to by the
    //        data parameter is expected to be the same as the 
    //        packetSize field of this UdpCustomPacketHeader.
    std::unique_ptr<char[]> createUdpPacket(char* data) const;
};

namespace UdpCustomPacket
{
    // Total size of header: 3 * 4 bytes + 2 * 2 bytes = 16 bytes
    // 
    // Header contains 3 ints with 4 bytes each
    // and 2 ints with 2 bytes each
    const int headerSizeBytes = 16;
    
    // The maximum size of the data encapsulated within this
    // With the header, the maximum size should add up to 65,507 bytes
    const int32_t maxPacketSize = UDP_MAX_DATA_LENGTH - headerSizeBytes;

    // Retrieves the custom packet header from the raw UDP data
    // 
    // data - Pointer to the buffer containing the data that was
    //        received. The length in bytes of the buffer pointed
    //        to by the data parameter should be greater than or
    //        equal to the size of the custom packet header.
    UdpCustomPacketHeader getHeader(char* data);
}

#endif UDP_CUSTOM_PACKET_H
