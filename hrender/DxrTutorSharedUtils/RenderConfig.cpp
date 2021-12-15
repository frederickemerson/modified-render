#include "RenderConfig.h"


std::vector<RenderConfig::Config> RenderConfig::mConfig(2);
void* RenderConfig::loc = 0;

void RenderConfig::setConfiguration(std::vector<HrenderType> orderedTypes) {
    // reset current configuration
    mConfig.clear();

    // add each configuration
    for (const auto& hrenderType : orderedTypes) {
        // get corresponding string name
        std::string name = RenderConfig::hrenderTypeToString(hrenderType);

        Config config = { hrenderType, name, -1, nullptr };
        mConfig.emplace_back(config);
    }
}

std::vector<RenderConfig::Config> RenderConfig::getConfig() {
    return RenderConfig::mConfig;
}

std::string RenderConfig::hrenderTypeToString(RenderConfig::HrenderType htype) {
    switch (htype) {

    case HrenderType::VisibilityBitmap: return "Visibility Bitmap";

    default: return "";
    }
}

std::string RenderConfig::print() {
    int i = 0;
    std::string out = std::to_string(mConfig.size());
    for (int i = 0; i < RenderConfig::mConfig.size(); i++) {
        out += "\n{" + std::to_string(i) + ": " 
            + RenderConfig::mConfig[i].name + ", "
            + std::to_string(RenderConfig::mConfig[i].resourceIndex) + ", "
            + std::to_string((uintptr_t)RenderConfig::mConfig[i].cpuLocation) + "}";
    }
    return out;
}