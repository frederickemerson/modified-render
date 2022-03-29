#include "Regression.h"

int Regression::frameNumber = 0;
std::string Regression::fileName = "RegressionData.txt";
Regression::CameraCoordinates Regression::camData = { 0,0,0,0,0,0,0,0,0 };
std::queue<Regression::CameraCoordinates> Regression::queueCameraChangeData;
std::queue<int*> Regression::queueNonSeqFrame;

void Regression::initialise()
{
    frameNumber = 0;
    fileName = "RegressionData.txt";
    camData = { 0,0,0,0,0,0,0,0,0 };
}

int Regression::calculateLoss(int* sequentialFrame, int* nonSequentialFrame) {
    int loss = 0;

    for (int i = 0; i < 1920 * 1080 / 7; i++) {
        int A = sequentialFrame[i * 7];
        int B = nonSequentialFrame[i * 7];
        int beforeLoss = loss;
        //char msg[50];

        for (int j = 0; j < 3; j++) {
            if (((A >> j) & 1) != ((B >> j) & 1)) {
                loss++;
            }
        }
        //if (loss > beforeLoss) {
        //    sprintf(msg, "i: %d, loss: %d, beforeLoss: %d\n", i, loss, beforeLoss);
        //    OutputDebugStringA(msg);
        //}
    }
    return loss;
}

void Regression::addCamera(float cameraData[9])
{
}

void Regression::calculate(float cameraData[9], float networkPing) {
    //int loss = calculateLoss();
    //storeEntry(loss, cameraData, networkPing);
}

void Regression::storeEntry(int loss, float cameraData[9], float networkPing)
{
    std::ofstream myfile(fileName);
    if (myfile.is_open())
    {
        myfile << loss << " ";
        for (int i = 0; i < 9; i++) {
            myfile << cameraData[i] << " ";
        }
        myfile << networkPing << "\n";
        myfile.close();
    }
}

void Regression::addCamera(float UX, float UY, float UZ, float VX, float VY, float VZ, float WX, float WY, float WZ)
{
    Regression::CameraCoordinates dif = {
        std::abs(UX - camData.UX),
        std::abs(UY - camData.UY),
        std::abs(UZ - camData.UZ),
        std::abs(VX - camData.VX),
        std::abs(VY - camData.VY),
        std::abs(VZ - camData.VZ),
        std::abs(WX - camData.WX),
        std::abs(WY - camData.WY),
        std::abs(WZ - camData.WZ) };

    queueCameraChangeData.push(dif);

    camData = { UX, UY, UZ, VX, VY, VZ, WX, WY, WZ };
}

void Regression::addNetworkPing(int frameIndex, float networkPing)
{
}

void Regression::addNonSeqFrame(int* nonSequentialFrame)
{
    queueNonSeqFrame.push(nonSequentialFrame);
}
