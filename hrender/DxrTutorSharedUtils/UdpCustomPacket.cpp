// A class to encapsulate a custom protocol built on top of UDP

#include "UdpCustomPacket.h"

UdpCustomPacketHeader::UdpCustomPacketHeader(int32_t expectedSequenceNumber):
    sequenceNumber(expectedSequenceNumber)
{}

UdpCustomPacketHeader::UdpCustomPacketHeader(int32_t seqNum, int32_t pktSize, uint8_t* data):
    sequenceNumber(seqNum), packetSize(pktSize), udpData(data)
{}

UdpCustomPacketHeader::UdpCustomPacketHeader(int32_t seqNum, int32_t pktSize, int32_t frmNum,
                                 int32_t numFrmPkts, int32_t tmStmp, uint8_t* data):
    sequenceNumber(seqNum),
    packetSize(pktSize),
    frameNumber(frmNum),
    numOfFramePackets(numFrmPkts),
    timestamp(tmStmp),
    udpData(data)
{}

UdpCustomPacketHeader::~UdpCustomPacketHeader()
{
    delete[] udpData;
}

UdpCustomPacketHeader::UdpCustomPacketHeader(UdpCustomPacketHeader&& ucp):
    sequenceNumber(ucp.sequenceNumber),
    packetSize(ucp.packetSize),
    frameNumber(ucp.frameNumber),
    numOfFramePackets(ucp.numOfFramePackets),
    timestamp(ucp.timestamp),
    udpData(ucp.udpData)
{
    ucp.sequenceNumber = -1;
    ucp.packetSize = 0;
    ucp.frameNumber = -1;
    ucp.numOfFramePackets = 0;
    ucp.timestamp = -1;
    ucp.udpData = nullptr;
}

UdpCustomPacketHeader& UdpCustomPacketHeader::operator=(UdpCustomPacketHeader&& ucp)
{
    this->sequenceNumber = ucp.sequenceNumber;
    this->packetSize = ucp.packetSize;
    this->frameNumber = ucp.frameNumber;
    this->numOfFramePackets = ucp.numOfFramePackets;
    this->timestamp = ucp.timestamp;
    this->udpData = ucp.udpData;

    ucp.sequenceNumber = -1;
    ucp.packetSize = 0;
    ucp.frameNumber = -1;
    ucp.numOfFramePackets = 0;
    ucp.timestamp = -1;
    ucp.udpData = nullptr;

    return *this;
}

// A helper function to add an int32_t to a char array at the specified offset
// Returns the offset for the next empty position of the char array
int addInt32ToCharPtr(int32_t data, std::unique_ptr<char[]>& array, int offset)
{
    int sizeOfData = 4;
    const uint8_t* dataBytes = reinterpret_cast<const uint8_t*>(&data);
    for (int i = 0; i < sizeOfData; i++)
    {
        array[i + offset] = static_cast<char>(dataBytes[i]);
    }
    return offset + sizeOfData;
}

std::unique_ptr<char[]> UdpCustomPacketHeader::createUdpPacket() const
{
    int32_t totalSize = UdpCustomPacketHeader::headerSizeBytes + packetSize;
    std::unique_ptr<char[]> udpPacket = std::make_unique<char[]>(totalSize);

    // Append header
    int offset = 0;
    offset = addInt32ToCharPtr(sequenceNumber, udpPacket, offset);
    offset = addInt32ToCharPtr(packetSize, udpPacket, offset);
    offset = addInt32ToCharPtr(frameNumber, udpPacket, offset);
    offset = addInt32ToCharPtr(numOfFramePackets, udpPacket, offset);
    offset = addInt32ToCharPtr(timestamp, udpPacket, offset);

    // Append data
    for (int i = 0; i < packetSize; i++)
    {
        udpPacket[i + offset] = udpData[i];
    }
    return udpPacket;
}

std::pair<int32_t, std::vector<UdpCustomPacketHeader>> UdpCustomPacketHeader::splitPacket() const
{
    int32_t currentSeqNum = sequenceNumber;
    std::vector<UdpCustomPacketHeader> splitPackets{};

    int numberOfNewPackets = packetSize / UdpCustomPacketHeader::maxPacketSize +
                             ((packetSize % UdpCustomPacketHeader::maxPacketSize > 0) ? 1 : 0);
    int newNumOfFramePackets = numOfFramePackets - 1 + numberOfNewPackets;

    int currentIndex = 0;
    for (int32_t amountLeft = packetSize; amountLeft > 0; amountLeft -= maxPacketSize)
    {
        int32_t size = std::min(amountLeft, UdpCustomPacketHeader::maxPacketSize);
        uint8_t* data = new uint8_t[size];
        for (int i = 0; i < size; i++)
        {
            data[i] = udpData[currentIndex];
            currentIndex++;
        }
        splitPackets.emplace_back(currentSeqNum, size, frameNumber,
                                  newNumOfFramePackets, timestamp, data);
        currentSeqNum++;
    }

    return std::pair<int32_t, std::vector<UdpCustomPacketHeader>>(currentSeqNum, std::move(splitPackets));
}

char* UdpCustomPacketHeader::getUdpDataPointer() const
{
    return reinterpret_cast<char*>(udpData);
}

void UdpCustomPacketHeader::setDataPointer(uint8_t* data)
{
    udpData = data;
}

void UdpCustomPacketHeader::copyInto(uint8_t* dataOut) const
{
    for (int i = 0; i < packetSize; i++)
    {
        dataOut[i] = udpData[i];
    }
}

void UdpCustomPacketHeader::copyIntoAndRelease(UdpCustomPacketHeader& copy)
{    
    // Free the data pointer originally used in the copy
    delete[] copy.udpData;

    copy.sequenceNumber = this->sequenceNumber;
    copy.packetSize = this->packetSize;
    copy.frameNumber = this->frameNumber;
    copy.numOfFramePackets = this->numOfFramePackets;
    copy.timestamp = this->timestamp;
    copy.udpData = this->udpData;

    this->sequenceNumber = -1;
    this->packetSize = 0;
    this->frameNumber = -1;
    this->numOfFramePackets = 0;
    this->timestamp = -1;
    this->udpData = nullptr;
}

uint8_t* UdpCustomPacketHeader::releaseDataPointer()
{
    uint8_t* ptr = udpData;
    udpData = nullptr;
    return ptr;
}
