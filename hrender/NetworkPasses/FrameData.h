#pragma once

// Stores the metadata for a specific frame
typedef struct FrameData
{
    int frameSize;      // Total size of the frame in bytes
    int frameNumber;    // Number associated to the current frame
    int timestamp;      // Time offset from the start in milliseconds
} FrameData;