#pragma once

#include <vector>
#include <string>
#include <map>
#include <iostream>

class RenderConfig {
public:
    enum class BufferType {
        VisibilityBitmap
    };

    struct Config {
        BufferType type;
        std::string name; // name in ResourceManager
        int resourceIndex; // index in ResourceManager
        void* networkPassOutputLocation;
        void* compressionPassOutputLocation;
        void* compressionPassOutputLocation2;
        int compressedSize;
        int compressedSize2;
    };

    static void setConfiguration(std::vector<BufferType> orderedTypes);
    static std::string print();
    static int getTotalSize();
    static int BufferTypeToSize(RenderConfig::BufferType htype);

    static std::vector<Config> getConfig();
    static std::vector<Config> mConfig;

protected:
    static int totalSize;
    static std::string BufferTypeToString(BufferType htype);
};
