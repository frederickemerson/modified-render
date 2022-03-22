# Hardware Ray-Tracing Assisted Hybrid Rendering Pipeline for Games
This project creates a distributed rendering algorithm using two devices across the network that combines rasterization and raytracing techniques. This is done to enable the better visual quality that raytracing brings compared to traditional rasterization techniques, to devices that cannot support hardware accelerated raytracing, by offloading the raytracing steps of the real-time rendering algorithm to a server. For now, our implementation has a direct-lighting shading algorithm that makes use of distributed rendering and raytracing. Currently, our implementation supports static scenes that allows user input to move the camera. The current implementation’s visual quality has need of improvement, requiring the addition of the suggested shading algorithm augmentations (in the FYP report) before it is ready for production, and the framerate is low at 7-8 fps, and requires the addition of H.264 compression and possibly changing from TCP to UDP to overcome the current network bottleneck. In the [Final report](Final_Report.pdf), we discuss the areas of improvement and future work. 

Before working on the code base, please read through the following sections in the README (that you are currently reading), followed by the [Final report](Final_Report.pdf) which will give a high level idea of the project. It is then recommended to look at [Chris Wyman's DXR tutorial](http://cwyman.org/code/dxrTutors/dxr_tutors.md.html) to get an understanding of the code base and the rendering pipeline in general, before finally going through the [Developer Guide](docs/DeveloperGuide.md) for a more in-depth walkthrough on navigating pipeline we have implemented.

## Important Files/Folders
- [Falcor](Falcor) 
  - Stores the Falcor 4.3 files from [NVIDIA's github page](https://github.com/NVIDIAGameWorks/Falcor), with modifications. 
  - Before modifying this with the latest version of Falcor, please check the [Developer Guide](docs/DeveloperGuide.md) on changes that need to be made to maintain compatibility
  - The project is accessed through the Falcor.sln file in this directory
- [hrender](hrender)
  - Stores our custom passes, as well as framework and passes from the [DXR tutorials](http://cwyman.org/code/dxrTutors/dxr_tutors.md.html) which have been modified for our purposes
- [Final report](Final_Report.pdf) and [slides](Final_Presentation_Slides_and_Script.pptx)
  - Explains the high level architecture of our rendering pipeline
- [Developer Guide](docs/DeveloperGuide.md)
  - Talks about the codebase modifications in more detail, 
- [Future work recommendations](Future_Work_Recommendations.md)
  - Talks about features that are recommended for development

## Prerequisites
(Taken from [Falcor/README.md](Falcor/README.md))
- Windows 10 version 2004 (May 2020 Update) or newer
- Visual Studio 2019
- [Windows 10 SDK (10.0.19041.0) for Windows 10, version 2004](https://developer.microsoft.com/en-us/windows/downloads/windows-10-sdk/)
- A GPU which supports DirectX Raytracing, such as the NVIDIA Titan V or GeForce RTX (make sure you have the latest driver)

Optional:
- Windows 10 Graphics Tools. To run DirectX 12 applications with the debug layer enabled, you must install this. There are two ways to install it:
    - Click the Windows button and type `Optional Features`, in the window that opens click `Add a feature` and select `Graphics Tools`.
    - Download an offline package from [here](https://docs.microsoft.com/en-us/windows-hardware/test/hlk/windows-hardware-lab-kit#supplemental-content-for-graphics-media-and-mean-time-between-failures-mtbf-tests). Choose a ZIP file that matches the OS version you are using (not the SDK version used for building Falcor). The ZIP includes a document which explains how to install the graphics tools.
- NVAPI (see below)

## NVAPI installation
After cloning the repository, head over to https://developer.nvidia.com/nvapi and download the latest version of NVAPI (this build is tested against version R470).
Create the folder `.packman` in `Source/Externals/`. Extract the contents of the zip file into the folder and rename `R470-developer` to `nvapi`.

Finally, set `_ENABLE_NVAPI` to `1` in `Source/Falcor/Core/FalcorConfig.h`

## Usage
Clone the repository or download it as a zip file. The solution file that contains the project is in the `./Falcor` folder, `./Falcor/Falcor.sln`. 

After opening the solution, in the solution explorer, right click and set `hrender` as the startup project.

![Set as startup project](docs/images/set_as_startup.png)

Set the debug mode to either DebugD3D12 or ReleaseD3D12.

![Debug mode](docs/images/d3d12_mode.png)

Right click the solution name in the solution explorer and build the solution.

![Build solution](docs/images/build_solution.png)

### Running on a single machine
To run it on a single machine for debugging, ensure go to the project properties, and under Debugging > Command Arguments, put the arguments "no-compression" and "debug".

![Properties](docs/images/properties.png)
![Debug properties](docs/images/no-compression_debug.png)

Then simply run the program.

### Running the distributed algorithm on two machines
To run the distributed pipeline, have the program set up on two separate machines. On one machine, the command arguments should be `no-compression server`, and the other should be `no-compression client` (you may exlude "no-compression", which will enable LZO compresion, which may worsen performance). For now, we have not implemented running both the server and client on a single device.

To enable use of communication over UDP, add `udp` to to command arguments. For example, the server
should be `no-compression udp server`, and the other should be `no-compression udp client`.

On the client PC, `hrender.cpp` must specify the server's IP address for TCP communication under the line
`ResourceManager::mNetworkManager->SetUpClient("192.168.1.111", DEFAULT_PORT);`, or for UDP, under the line
`ResourceManager::mNetworkManager->SetUpClientUdp("172.26.186.144", DEFAULT_PORT_UDP);`
You may change the value of `DEFAULT_PORT` or `DEFAULT_PORT_UDP` for the program to communicate on a different port.

The server's IP address can be acquired using command prompt and `ipconfig`.

The server should start running first, then the client. A debug message will be shown 
in the server when it is waiting for the client, if you run the program with the
help of Visual Studio:
`= Pre-Falcor Init - Trying to listen for client width/height... =========`

An example of what the server and client will see is available on the [demonstration video](Demonstration_Video.mkv).

Currently, we do not have a way to dynamically load a new scene, so to use a different scene, under any occurrences in the code of `setDefaultSceneName`, ensure that the arguments to the function are set to the scene you would like to use. For now, this is in `JitteredGBufferPass.cpp` and `VisibilityPass.cpp` - set the `kDefaultScene` values in these files to point to the path of the scene you would like to use.
