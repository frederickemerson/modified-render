#pragma once
#include <iostream>
#include <fstream>
#include <string>
#include <queue>

/**
 * Transfer data from server to client or client to server
 * based on the configuration setting.
 */

class Regression {

public:
    struct CameraCoordinates {
        float UX;
        float UY;
        float UZ;
        float VX;
        float VY;
        float VZ;
        float WX;
        float WY;
        float WZ;
    };

    static void initialise();
    // called when network sends camera data
    static void addCamera(float cameraData[9]);
    static void addCamera(float UX, float UY, float UZ, float VX, float VY, float VZ, float WX, float WY, float WZ);

    static void addNetworkPing(int frameIndex, float networkPing);

    // called when CPU GPU memory transfer occurs
    static void addNonSeqFrame(int* nonSequentialFrame);

    void calculate(float cameraData[9], float networkPing);
    static int calculateLoss(int* sequentialFrame, int* nonSequentialFrame);
    static void storeEntry(int loss, float cameraData[9], float networkPing);

    char msg[100];

protected:
    static std::queue<CameraCoordinates> queueCameraChangeData;
    static std::queue<int*> queueNonSeqFrame;

    // camera data
    static CameraCoordinates camData;

    static std::string fileName;
    static int frameNumber;
    const static int interval = 3;
};