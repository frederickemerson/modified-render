// A class to encapsulate a custom protocol built on top of UDP

#ifndef UDP_CUSTOM_PACKET_H
#define UDP_CUSTOM_PACKET_H

#include <memory>
#include <vector>

class UdpCustomPacketHeader {
public:
    // Header size in bytes
    // Header contains 5 ints with 4 bytes each
    const static int headerSizeBytes = 20;
    
    // The maximum size of the data encapsulated within this
    // With the header, the maximum size should add up to 65,507 bytes
    const static int32_t maxPacketSize = 65487;

    // ======================= HEADER FIELDS =======================
    // Total size of header: 5 * 4 bytes = 20 bytes
    
    // 1) Sequence number of this packet
    int32_t sequenceNumber = -1;
    // 2) Packet size in bytes, excluding header size
    int32_t packetSize = 0;
    // 3) Frame number which this packet belongs to
    int32_t frameNumber = -1;
    // 4) Number of packets that belong to the same frame
    int32_t numOfFramePackets = 0;
    // 5) Timestamp for when the frame should be played
    int32_t timestamp = -1;

    // =========================== DATA ============================
    // Raw UDP packet, an array of bytes
    uint8_t* udpData = nullptr;

    // Initialise a packet with just the sequence number
    UdpCustomPacketHeader(int32_t expectedSequenceNumber);
    // Initialise a packet with without frame information
    UdpCustomPacketHeader(int32_t seqNum, int32_t pktSize, uint8_t* data);
    // Initialise a packet with all the fields
    UdpCustomPacketHeader(int32_t seqNum, int32_t pktSize, int32_t frmNum,
                    int32_t numFrmPkts, int32_t tmStmp, uint8_t* data);
    UdpCustomPacketHeader() = delete;
    UdpCustomPacketHeader(const UdpCustomPacketHeader&) = delete;
    UdpCustomPacketHeader& operator=(const UdpCustomPacketHeader&) = delete;
    UdpCustomPacketHeader(UdpCustomPacketHeader&&);
    UdpCustomPacketHeader& operator=(UdpCustomPacketHeader&&);
    ~UdpCustomPacketHeader();

    // Create a UDP packet to send
    std::unique_ptr<char[]> createUdpPacket() const;

    // Splits the packet and returns a pair of
    // the next sequence number and the packets
    // 
    // Packets returned are guaranteed to be smaller
    // or equal to the max packet size
    std::pair<int32_t, std::vector<UdpCustomPacketHeader>> splitPacket() const;

    // Get the address of the packet data as a char pointer
    char* getUdpDataPointer() const;

    // Sets the udpData of this UdpCustomPacket
    void setDataPointer(uint8_t* data);

    // Copies the data from this packet into another array
    // Note: User must ensure that the array has enough
    //       allocated space!
    void copyInto(uint8_t* dataOut) const;

    // Copies the data from this packet into another UdpCustomPacket
    // and releases the pointer that is held by this packet
    //
    // Note: This calls delete[] on the data pointer in the copy
    void copyIntoAndRelease(UdpCustomPacketHeader& copy);

    // Returns the data pointer and sets the pointer
    // of this UdpCustomPacket to nullptr
    // 
    // Used to ensure that this UdpCustomPacket does not
    // delete the pointer when it is deleted
    uint8_t* releaseDataPointer();
};

#endif UDP_CUSTOM_PACKET_H
