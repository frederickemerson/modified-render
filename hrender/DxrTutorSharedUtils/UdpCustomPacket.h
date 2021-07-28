// A class to encapsulate a custom protocol built on top of UDP

#ifndef UDP_CUSTOM_PACKET_H
#define UDP_CUSTOM_PACKET_H

#include <memory>
#include <vector>

class UdpCustomPacket {
public:
    // Header size in bytes
    // Header contains two ints, sequence number and packet size
    const static int headerSizeBytes = 8;
    
    // The maximum size of the data encapsulated within this
    // With the header, the maximum size should add up to 65,507 bytes
    const static int32_t maxPacketSize = 65499;


    // Sequence number of this packet
    int32_t sequenceNumber;
    // Packet size in bytes, excluding header size
    int32_t packetSize;

    // Raw UDP packet, an array of bytes
    uint8_t* udpData;

    // Initialise a packet with just the sequence number
    UdpCustomPacket(int32_t expectedSequenceNumber);
    // Initialise a packet with all the fields
    UdpCustomPacket(int32_t seqNum, int32_t pktSize, uint8_t* data);
    UdpCustomPacket() = delete;
    UdpCustomPacket(const UdpCustomPacket&) = delete;
    UdpCustomPacket& operator=(const UdpCustomPacket&) = delete;
    UdpCustomPacket(UdpCustomPacket&&);
    UdpCustomPacket& operator=(UdpCustomPacket&&);
    ~UdpCustomPacket();

    // Create a UDP packet to send
    std::unique_ptr<char[]> createUdpPacket() const;

    // Splits the packet and returns a pair of
    // the next sequence number and the packets
    // 
    // Packets returned are guaranteed to be smaller
    // or equal to the max packet size
    std::pair<int32_t, std::vector<UdpCustomPacket>> splitPacket() const;

    // Get the address of the packet data as a char pointer
    char* getUdpDataPointer() const;

    // Sets the udpData of this UdpCustomPacket
    void setDataPointer(uint8_t* data);

    // Copies the data from this packet into another array
    // Note: User must ensure that the array has enough
    //       allocated space!
    void copyInto(uint8_t* dataOut);

    // Returns the data pointer and sets the pointer
    // of this UdpCustomPacket to nullptr
    // 
    // Used to ensure that this UdpCustomPacket does not
    // delete the pointer when it is deleted
    uint8_t* releaseDataPointer();
};

#endif UDP_CUSTOM_PACKET_H
