#pragma once

#include <vector>
#include <string>
#include <map>
#include <iostream>

class RenderConfig {
public:
    enum class HrenderType {
        VisibilityBitmap
    };

    struct Config {
        HrenderType type;
        std::string name; // name in ResourceManager
        int resourceIndex; // index in ResourceManager
        void* cpuLocation;
    };

    static void setConfiguration(std::vector<HrenderType> orderedTypes);
    static void* loc;
    static std::vector<Config> getConfig();
    static std::vector<Config> mConfig;
    static std::string print();
protected:
    static std::string hrenderTypeToString(HrenderType htype);
};
