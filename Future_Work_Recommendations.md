# Future Work Recommendations
The following are the recommendations for future work. Although these are discussed extensively in the [Final Report](Final_Report.pdf), and it is recommended to read through it, this document summarizes them as well.

## New features to implement (no particular order)
- Compression
  - Implementation of H.264/AVC encoding to compress the textures sent over the network (for now, just the visibility bitmap)
- Network improvements
  - Implementation of UDP instead of TCP for the network manager
    - Even with a single hop router, packets will be dropped eventually, so a robust implementation is necessary. There is a branch (server-g) that has a basic implementation of UDP that you may reference, although it is not up to date, but the network manager code is still usable
  - Make it possible to reuse frames if no changes were made - more sophisticated messaging between the client and network, as well as multithreading is necessary
- Visual quality
  - Make a rendering algorithm that supports global illumination and denoising
    - The code is actually available for these in a non-distributed context, please see [DxrTutorCommonPasses](hrender/DxrTutorCommonPasses) for global illumination passes that were available in the [DXR tutorial](http://cwyman.org/code/dxrTutors/dxr_tutors.md.html) and modified/fixed for our purposes, and [SVGFPasses](hrender/SVGFPasses) for denoiser passes that were taken from [NVIDIA's page](https://research.nvidia.com/publication/2017-07_Spatiotemporal-Variance-Guided-Filtering%3A) and upgraded to Falcor 4.3. 
    - The recommended architecture for use in the distributed pipeline is discussed in section 3.3 of the [Final Report](Final_Report.pdf).
  - Our current direct lighting pipeline supports up to 32 lights, either look into increasing this or take another approach.
- Scene management
  - We need to support animated scenes. 
  - We need to support sending scenes from the client to the server, instead of pre-loading the files with hard-coding the path of the scene in the code.
- Miscellaneous
  - We need to support resizing of the window, for now, after starting up the client/server, window size is fixed
  - Speeding up of the memory transfer from GPU to CPU is necessary. This was attempted with changes to `CopyContext.cpp/h`, `D3D12CopyContext.cpp` and `VKCopyContext.cpp` but was unsuccessful. Perhaps looking into CUDA pinned memory would be helpful? But I am unsure.
- In the far future, porting over to mobile is recommended.