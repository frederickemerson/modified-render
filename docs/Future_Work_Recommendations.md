# Future Work Recommendations

The following are some recommendations for future work and extensions.

## Compression

- Implementation of H.264/AVC encoding to temporally compress the textures sent over the network (for now, just the visibility bitmap). 
- Look into video streaming protocols like RTSP. We do not need control protocols for play/pause functions, but the other parts should be useful.

## Network
- Server-side Prediction
    - Client or server calculates a few potential camera orientations for subsequent frames
    - Server sends the client a few visibility buffers. These are expected to be highly similar and hence highly compressible
    - Client uses the closest visibility buffer to the actual camera orientation for client-side prediction in `PredictionPass`

## Visual Quality

- Make a rendering algorithm that supports **global illumination** and **denoising**
    - The code is actually available for these in a non-distributed context, please see [DxrTutorCommonPasses](../hrender/DxrTutorCommonPasses) for global illumination passes that were available in the [DXR Tutorial](http://cwyman.org/code/dxrTutors/dxr_tutors.md.html) and modified/fixed for our purposes, and [SVGFPasses](../hrender/SVGFPasses) for denoiser passes that were taken from [NVIDIA's webpage](https://research.nvidia.com/publication/2017-07_Spatiotemporal-Variance-Guided-Filtering%3A) and upgraded to Falcor 4.3. 
    - The recommended architecture to use in the distributed pipeline is discussed in section 3.3 of [Louiz's Final Report](FYP_Final_Report_Louiz.pdf).
  - Our current direct lighting pipeline supports up to 32 lights, either look into increasing this or take another approach.

## Scene Management

- We need to support animated scenes.
- We need to support sending scene changes from the client to the server, instead of pre-loading the files and hard-coding the path of the scene in the code.

## Miscellaneous

- Is there any benefit to sending visibility data back to the client for rasterisation? Rendering the full image on the server with ray tracing and sending it to the client may be more efficient, due to the coherence and cache-friendliness of the visibility buffer.
- We need to support resizing the window. For now, after starting up the client/server, the window size is fixed.
- Speeding up the memory transfer from GPU to CPU is necessary. This was attempted with changes to `CopyContext.cpp/h`, `D3D12CopyContext.cpp` and `VKCopyContext.cpp` but was unsuccessful. Perhaps looking into CUDA pinned memory would be helpful?
- In the far future, porting over to and testing on mobile is recommended.
