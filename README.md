# hrender
Simple Lightweight Hybrid Rendering Engine

## Prerequisites
- Windows 10 version 1809 or newer
- Visual Studio 2019
- [Microsoft Windows SDK version 1903 (10.0.18362.1)](https://developer.microsoft.com/en-us/windows/downloads/sdk-archive)

Optional:
- A GPU which supports DirectX Raytracing, such as the NVIDIA Titan V or GeForce RTX (make sure you have the latest driver)
- Windows 10 Graphics Tools. To run DirectX 12 applications with the debug layer enabled, you must install this. There are two ways to install it:
    - Click the Windows button and type `Optional Features`, in the window that opens click `Add a feature` and select `Graphics Tools`.
    - Download an offline package from [here](https://docs.microsoft.com/en-us/windows-hardware/test/hlk/windows-hardware-lab-kit#supplemental-content-for-graphics-media-and-mean-time-between-failures-mtbf-tests). Choose a ZIP file that matches the OS version you are using (not the SDK version used for building Falcor). The ZIP includes a document which explains how to install the graphics tools.
- NVAPI (see below)

## NVAPI installation
After cloning the repository, head over to https://developer.nvidia.com/nvapi and download the latest version of NVAPI (this build is tested against version R435).
Extract the content of the zip file into `Source/Externals/.packman/` and rename `R435-developer` to `NVAPI`.

Finally, set `_ENABLE_NVAPI` to `true` in `Source/Falcor/Core/FalcorConfig.h`