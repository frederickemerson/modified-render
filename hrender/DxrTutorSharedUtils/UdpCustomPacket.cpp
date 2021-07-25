// A class to encapsulate a custom protocol built on top of UDP

#include "UdpCustomPacket.h"

UdpCustomPacket::UdpCustomPacket(int32_t expectedSequenceNumber):
    sequenceNumber(expectedSequenceNumber), packetSize(0), udpData(nullptr)
{}

UdpCustomPacket::UdpCustomPacket(int32_t seqNum, int32_t pktSize, uint8_t* data):
    sequenceNumber(seqNum), packetSize(pktSize), udpData(data)
{ }

UdpCustomPacket::~UdpCustomPacket()
{
    delete[] udpData;
}

UdpCustomPacket::UdpCustomPacket(UdpCustomPacket&& ucp):
    sequenceNumber(ucp.sequenceNumber), packetSize(ucp.packetSize), udpData(ucp.udpData)
{
    ucp.sequenceNumber = -1;
    ucp.packetSize = 0;
    ucp.udpData = nullptr;
}

UdpCustomPacket& UdpCustomPacket::operator=(UdpCustomPacket&& ucp)
{
    this->sequenceNumber = ucp.sequenceNumber;
    this->packetSize = ucp.packetSize;
    this->udpData = ucp.udpData;
    ucp.sequenceNumber = -1;
    ucp.packetSize = 0;
    ucp.udpData = nullptr;
    return *this;
}

std::unique_ptr<char[]> UdpCustomPacket::createUdpPacket()
{
    int32_t totalSize = UdpCustomPacket::headerSizeBytes + packetSize;
    std::unique_ptr<char[]> udpPacket = std::make_unique<char[]>(totalSize);\

    // Append header
    int sizeOfSeqNum = 4;
    uint8_t* seqNum = reinterpret_cast<uint8_t*>(&sequenceNumber);
    int i = 0;
    for (i = 0; i < sizeOfSeqNum; i++)
    {
        udpPacket[i] = static_cast<char>(seqNum[i]);
    }
    uint8_t* pktSize = reinterpret_cast<uint8_t*>(&packetSize);
    for (i = sizeOfSeqNum; i < UdpCustomPacket::headerSizeBytes; i++)
    {
        udpPacket[i] = static_cast<char>(pktSize[i - sizeOfSeqNum]);
    }

    // Append data
    int j = 0;
    for (j = 0; j < packetSize; j++)
    {
        udpPacket[j + UdpCustomPacket::headerSizeBytes] = udpData[j];
    }
    return udpPacket;
}

std::pair<int32_t, std::vector<UdpCustomPacket>> UdpCustomPacket::splitPacket()
{
    int32_t currentSeqNum = sequenceNumber;
    std::vector<UdpCustomPacket> splitPackets{};
    int currentIndex = 0;

    for (int32_t amountLeft = packetSize; amountLeft > 0; amountLeft -= maxPacketSize)
    {
        int32_t size = amountLeft > maxPacketSize ? maxPacketSize : amountLeft;
        uint8_t* data = new uint8_t[size];
        for (int i = 0; i < size; i++)
        {
            data[i] = udpData[currentIndex];
            currentIndex++;
        }
        splitPackets.emplace_back(currentSeqNum, size, data);
        currentSeqNum++;
    }

    return std::pair<int32_t, std::vector<UdpCustomPacket>>(currentSeqNum, std::move(splitPackets));
}

char* UdpCustomPacket::getUdpDataPointer()
{
    return reinterpret_cast<char*>(udpData);
}
