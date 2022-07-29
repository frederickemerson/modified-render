#include "RenderConfig.h"

#define VIS_TEX_LEN 8294400 // 4 * 1920 * 1080
#define REF_TEX_LEN 8294400 // 4 * 1920 * 1080

std::vector<RenderConfig::Config> RenderConfig::mConfig(1);
int RenderConfig::totalSize = 0;

void RenderConfig::setConfiguration(std::vector<BufferType> orderedTypes) {
    // reset current configuration
    mConfig.clear();

    // add each configuration
    for (const auto& bufferType : orderedTypes) {
        // get corresponding string name
        std::string bufferName = BufferTypeToString(bufferType);
        int bufferSize = BufferTypeToSize(bufferType);

        Config config = { bufferType, bufferName, -1, bufferSize, 0 };
        mConfig.emplace_back(config);

        totalSize += bufferSize;
    }
}

std::vector<RenderConfig::Config> RenderConfig::getConfig() {
    return RenderConfig::mConfig;
}

std::string RenderConfig::BufferTypeToString(RenderConfig::BufferType htype) {
    switch (htype) {
        case BufferType::VisibilityBitmap: return "Visibility Bitmap";
        case BufferType::SRTReflection: return "Server Ray Tracing Reflection";
    }

    return "";
}

int RenderConfig::BufferTypeToSize(RenderConfig::BufferType htype) {
    switch (htype) {
        case BufferType::VisibilityBitmap: return VIS_TEX_LEN;
        case BufferType::SRTReflection: return REF_TEX_LEN;
    }

    return 0;
}

std::string RenderConfig::print() {
    int i = 0;
    std::string out = std::to_string(mConfig.size());
    for (int i = 0; i < RenderConfig::mConfig.size(); i++) {
        out += "\n{" + std::to_string(i) + ": " 
            + RenderConfig::mConfig[i].name + ", "
            + std::to_string(RenderConfig::mConfig[i].resourceIndex) + "}";
    }
    return out;
}

int RenderConfig::getTotalSize() {
    return RenderConfig::totalSize;
}
