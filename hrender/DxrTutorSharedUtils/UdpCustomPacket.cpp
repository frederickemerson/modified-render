// A class to encapsulate a custom protocol built on top of UDP

#include "UdpCustomPacket.h"

// A helper function to add some data to a char array at the specified offset
// Returns the offset for the next empty position of the char array
int addToCharPtr(const uint8_t* dataBytes, int sizeOfData, std::unique_ptr<char[]>& array, int offset)
{
    for (int i = 0; i < sizeOfData; i++)
    {
        array[i + offset] = static_cast<char>(dataBytes[i]);
    }
    return offset + sizeOfData;
}

// Helper function for adding int32_t to a char array
int addInt32ToCharPtr(const int32_t* data, std::unique_ptr<char[]>& array, int offset)
{
    const uint8_t* int32Bytes = reinterpret_cast<const uint8_t*>(data);
    return addToCharPtr(int32Bytes, 4, array, offset);
}

// Same as addInt32ToCharPtr, but for int16_t
int addInt16ToCharPtr(const uint16_t* data, std::unique_ptr<char[]>& array, int offset)
{
    const uint8_t* int16Bytes = reinterpret_cast<const uint8_t*>(data);
    return addToCharPtr(int16Bytes, 2, array, offset);
}

UdpCustomPacketHeader::UdpCustomPacketHeader(int32_t seqNum, uint16_t dtSize):
    sequenceNumber(seqNum), dataSize(dtSize)
{}

UdpCustomPacketHeader::UdpCustomPacketHeader(int32_t seqNum, uint16_t dtSize, int32_t frmNum,
                                             uint16_t numFrmPkts, int32_t tmStmp):
    sequenceNumber(seqNum),
    dataSize(dtSize),
    frameNumber(frmNum),
    numOfFramePackets(numFrmPkts),
    timestamp(tmStmp)
{}

std::unique_ptr<char[]> UdpCustomPacketHeader::createUdpPacket(char* data) const
{
    int32_t totalSize = UdpCustomPacket::headerSizeBytes + dataSize;
    std::unique_ptr<char[]> udpPacket = std::make_unique<char[]>(totalSize);

    // Append header
    int offset = 0;
    offset = addInt32ToCharPtr(&sequenceNumber, udpPacket, offset);
    offset = addInt32ToCharPtr(&frameNumber, udpPacket, offset);
    offset = addInt16ToCharPtr(&dataSize, udpPacket, offset);
    offset = addInt16ToCharPtr(&numOfFramePackets, udpPacket, offset);
    offset = addInt32ToCharPtr(&timestamp, udpPacket, offset);

    // Append data
    for (int i = 0; i < dataSize; i++)
    {
        udpPacket[i + offset] = data[i];
    }
    return udpPacket;
}

UdpCustomPacketHeader UdpCustomPacket::getHeader(char* data)
{
    int32_t* headerData = reinterpret_cast<int32_t*>(data);
    uint16_t* smallFields = reinterpret_cast<uint16_t*>(&headerData[2]);

    int32_t seqNum = headerData[0];
    uint16_t dataSize = smallFields[0];
    int32_t frameNum = headerData[1];
    uint16_t numFramePkts = smallFields[1];
    int32_t timestamp = headerData[3];

    return UdpCustomPacketHeader(seqNum, dataSize, frameNum, numFramePkts, timestamp);
}
