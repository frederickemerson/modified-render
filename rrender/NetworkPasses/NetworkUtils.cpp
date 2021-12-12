#include "NetworkUtils.h"

std::chrono::milliseconds getCurrentTime()
{
    return std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::system_clock::now().time_since_epoch()
    );
}

double getFps(std::chrono::duration<double> timeForOneFrame)
{
    return std::chrono::duration<double>(1.0f) / timeForOneFrame;
}
