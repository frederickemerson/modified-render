// A class to encapsulate a custom protocol built on top of UDP

#include "UdpCustomPacket.h"

UdpCustomPacket::UdpCustomPacket(int32_t expectedSequenceNumber):
    sequenceNumber(expectedSequenceNumber)
{}

UdpCustomPacket::UdpCustomPacket(int32_t seqNum, int32_t pktSize, uint8_t* data):
    sequenceNumber(seqNum), packetSize(pktSize), udpData(data)
{}

UdpCustomPacket::UdpCustomPacket(int32_t seqNum, int32_t pktSize, int32_t frmNum,
                                 int32_t numFrmPkts, int32_t tmStmp, uint8_t* data):
    sequenceNumber(seqNum),
    packetSize(pktSize),
    frameNumber(frmNum),
    numOfFramePackets(numFrmPkts),
    timestamp(tmStmp),
    udpData(data)
{}

UdpCustomPacket::~UdpCustomPacket()
{
    delete[] udpData;
}

UdpCustomPacket::UdpCustomPacket(UdpCustomPacket&& ucp):
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

UdpCustomPacket& UdpCustomPacket::operator=(UdpCustomPacket&& ucp)
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

std::unique_ptr<char[]> UdpCustomPacket::createUdpPacket() const
{
    int32_t totalSize = UdpCustomPacket::headerSizeBytes + packetSize;
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

std::pair<int32_t, std::vector<UdpCustomPacket>> UdpCustomPacket::splitPacket() const
{
    int32_t currentSeqNum = sequenceNumber;
    std::vector<UdpCustomPacket> splitPackets{};

    int numberOfNewPackets = packetSize / UdpCustomPacket::maxPacketSize +
                             ((packetSize % UdpCustomPacket::maxPacketSize > 0) ? 1 : 0);
    int newNumOfFramePackets = numOfFramePackets - 1 + numberOfNewPackets;

    int currentIndex = 0;
    for (int32_t amountLeft = packetSize; amountLeft > 0; amountLeft -= maxPacketSize)
    {
        int32_t size = std::min(amountLeft, UdpCustomPacket::maxPacketSize);
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

    return std::pair<int32_t, std::vector<UdpCustomPacket>>(currentSeqNum, std::move(splitPackets));
}

char* UdpCustomPacket::getUdpDataPointer() const
{
    return reinterpret_cast<char*>(udpData);
}

void UdpCustomPacket::setDataPointer(uint8_t* data)
{
    udpData = data;
}

void UdpCustomPacket::copyInto(uint8_t* dataOut) const
{
    for (int i = 0; i < packetSize; i++)
    {
        dataOut[i] = udpData[i];
    }
}

uint8_t* UdpCustomPacket::releaseDataPointer()
{
    uint8_t* ptr = udpData;
    udpData = nullptr;
    return ptr;
}
