// A file with helpful functions for neworking

#include <chrono>

// Gets the current time in milliseconds (since the epoch)
std::chrono::milliseconds getCurrentTime();

// Gets the number of frames per second, given the time of one frame
double getFps(std::chrono::duration<double> timeForOneFrame);
