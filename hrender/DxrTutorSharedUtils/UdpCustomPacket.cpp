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
int addInt32ToCharPtr(int32_t data, std::unique_ptr<char[]>& array, int offset)
{
    const uint8_t* int32Bytes = reinterpret_cast<const uint8_t*>(&data);
    return addToCharPtr(int32Bytes, 4, array, offset);
}

// Same as addInt32ToCharPtr, but for int16_t
int addInt16ToCharPtr(int16_t data, std::unique_ptr<char[]>& array, int offset)
{
    const uint8_t* int16Bytes = reinterpret_cast<const uint8_t*>(&data);
    return addToCharPtr(int16Bytes, 2, array, offset);
}

UdpCustomPacketHeader::UdpCustomPacketHeader(int32_t expectedSequenceNumber):
    sequenceNumber(expectedSequenceNumber)
{}

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
    int32_t totalSize = UdpCustomPacketHeader::headerSizeBytes + dataSize;
    std::unique_ptr<char[]> udpPacket = std::make_unique<char[]>(totalSize);

    // Append header
    int offset = 0;
    offset = addInt32ToCharPtr(sequenceNumber, udpPacket, offset);
    offset = addInt32ToCharPtr(frameNumber, udpPacket, offset);
    offset = addInt16ToCharPtr(dataSize, udpPacket, offset);
    offset = addInt16ToCharPtr(numOfFramePackets, udpPacket, offset);
    offset = addInt32ToCharPtr(timestamp, udpPacket, offset);

    // Append data
    for (int i = 0; i < dataSize; i++)
    {
        udpPacket[i + offset] = data[i];
    }
    return udpPacket;
}

UdpCustomPacketHeader UdpCustomPacket::getHeader(char* data)
{
    int32_t* headerData = reinterpret_cast<int32_t*>(&data);
    uint16_t* smallFields = reinterpret_cast<uint16_t*>(&headerData[2]);

    int32_t seqNum = headerData[0];
    uint16_t dataSize = smallFields[0];
    int32_t frameNum = headerData[1];
    uint16_t numFramePkts = smallFields[1];
    int32_t timestamp = headerData[3];

    return UdpCustomPacketHeader(seqNum, dataSize, frameNum, numFramePkts, timestamp);
}

// std::pair<int32_t, std::vector<UdpCustomPacketHeader>> UdpCustomPacketHeader::splitPacket() const
// {
//     int32_t currentSeqNum = sequenceNumber;
//     std::vector<UdpCustomPacketHeader> splitPackets{};

//     int numberOfNewPackets = packetSize / UdpCustomPacketHeader::maxPacketSize +
//                              ((packetSize % UdpCustomPacketHeader::maxPacketSize > 0) ? 1 : 0);
//     int newNumOfFramePackets = numOfFramePackets - 1 + numberOfNewPackets;

//     int currentIndex = 0;
//     for (int32_t amountLeft = packetSize; amountLeft > 0; amountLeft -= maxPacketSize)
//     {
//         int32_t size = std::min(amountLeft, UdpCustomPacketHeader::maxPacketSize);
//         uint8_t* data = new uint8_t[size];
//         for (int i = 0; i < size; i++)
//         {
//             data[i] = udpData[currentIndex];
//             currentIndex++;
//         }
//         splitPackets.emplace_back(currentSeqNum, size, frameNumber,
//                                   newNumOfFramePackets, timestamp, data);
//         currentSeqNum++;
//     }

//     return std::pair<int32_t, std::vector<UdpCustomPacketHeader>>(currentSeqNum, std::move(splitPackets));
// }

// char* UdpCustomPacketHeader::getUdpDataPointer() const
// {
//     return reinterpret_cast<char*>(udpData);
// }

// void UdpCustomPacketHeader::setDataPointer(uint8_t* data)
// {
//     udpData = data;
// }

// void UdpCustomPacketHeader::copyInto(uint8_t* dataOut) const
// {
//     for (int i = 0; i < packetSize; i++)
//     {
//         dataOut[i] = udpData[i];
//     }
// }

// void UdpCustomPacketHeader::copyIntoAndRelease(UdpCustomPacketHeader& copy)
// {    
//     // Free the data pointer originally used in the copy
//     delete[] copy.udpData;

//     copy.sequenceNumber = this->sequenceNumber;
//     copy.packetSize = this->packetSize;
//     copy.frameNumber = this->frameNumber;
//     copy.numOfFramePackets = this->numOfFramePackets;
//     copy.timestamp = this->timestamp;
//     copy.udpData = this->udpData;

//     this->sequenceNumber = -1;
//     this->packetSize = 0;
//     this->frameNumber = -1;
//     this->numOfFramePackets = 0;
//     this->timestamp = -1;
//     this->udpData = nullptr;
// }

// uint8_t* UdpCustomPacketHeader::releaseDataPointer()
// {
//     uint8_t* ptr = udpData;
//     udpData = nullptr;
//     return ptr;
// }
