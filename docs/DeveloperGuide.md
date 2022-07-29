# Developer Guide

This developer guide provides a summary of the passes and the classes in the solution.

You may refer to the [README](../README.md) for an explanation of how to install and run the program as well as links to other useful resources.

## Directory Structure

- `Falcor` - the Falcor 4.4 base code (with modifications mentioned below)
- `hrender` - The location of our source code for the pipeline
    - `Data` - Our default scenes and environment maps
    - `DxrTutorCommonPasses` - Render passes available in the DXR Tutorial (edited for our purposes)
    - `DxrTutorSharedUtils` - The main code for the rendering pipeline, and the abstract classes for render passes and shader programs.
        - See the **Shared Utils** section below for details about every class.
    - `DxrTutorTestPasses` - Test passes from the DXR Tutorial.
    - `Libraries` - Contains the header-only library for LZO compression. Currently unused.
    - `NetworkPasses` - Directory containing our distributed rendering passes.
        - See the **Network Pipeline** section below for details about every class.
    - `RasterizedPasses` - This contains a cascaded/omnidirectional mapped shadows implementation - for now, this is not used, but can be used in future. Not included in the solution right now.
    - `SVGFPasses` - This contains code from [NVIDIA's page](https://research.nvidia.com/publication/2017-07_Spatiotemporal-Variance-Guided-Filtering%3A), ported to Falcor 4.3 and abstracted. It is slightly buggy with walls aligned with the XZ plane. Not included in the solution right now.
    - `TestPasses` - More test passes for our use.
        - `ModulateAlbedoIllum` pass - this is used to debug the SVGF GBuffer and render passes. For example usage, please refer to the branch before the implementation of the network passes [here](https://github.com/loveandbejoyful/hrender/tree/pre-network).
            - Over there, you will find that hrender.cpp has some sequences of how you can make use of it, along with the SVGF passes, but currently, it will have a runtime error when using the SVGF passes as the shader does not get copied to the binary folder for some reason (probably an optimization issue). A quick fix is to copy the shader manually to the build folder that is specified in the error message where it says "failed to find include file 'svgfGBufData.hlsli'".
        - `DecodeGBufferPass` - This is used to preview the GBuffer's contents because we compacted the GBuffer as mentioned in Louiz's FYP report.

## Shared Utils (DxrTutorSharedUtils folder)
This folder should contain utility classes that may be used by multiple parts of the program.

#### `CircularBuffer`
Implementation of a generic circular buffer, used in `PredictionPass`.

#### `Compression`
A wrapper around the LZ4 compression library.

#### `CompressionPass`
A pass that employs the NVIDIA Video Codec SDK for compression. Currently unused.

#### `NvDecoder`
This is another class which deals with the NVIDIA Video Codec SDK.

#### `Regression`
A class used to take measurements of the pipeline.

#### `RenderConfig`
A class that allows easy reordering of the passes in the pipeline or changing of the pipeline variables.

#### `Semaphore`
An implementation of a binary semaphore (signal and wait functions). Used to communicate between the network threads and the rendering thread (on both the server and client).

#### `UdpCustomPacket`
Implementation of the custom UDP protocol for network communication. After the program has initialized, the protocol is used to send the camera data (from client to server) and the visibility buffer (from server to client).

### From DXR Tutorial
The classes below are all originally from the DXR Tutorial. All "Launch" classes are easy ways to load shaders for use in a rendering pass.

#### `FullscreenLaunch`
Takes in a single fragment shader. The vertex shader will be set to a default that does nothing.

#### `RasterLaunch`
Use this if you need both a vertex shader and a fragment shader.

#### `RayLaunch`
Use this for ray tracing shaders.

#### `RenderingPipeline`
This class encapsulates the entire rendering pipeline. It contains all of the rendering passes in order and can be executed to run the graphics program.

#### `RenderPass`
The abstract class for a rendering pass in the pipeline.

#### `ResourceManager`
A class that holds and supplies the resources needed for rendering such as framebuffer objects and textures.

#### `SceneLoaderWrapper`
Used by RenderingPipeline.

## Network Pipeline (NetworkPasses folder)
This folder should contain all classes related to network communications.

### Network Managers
This is where you will find code to set up the server and client for network transmission. The server and the client both contain two different threads for receiving and sending, and these are set up by the NetworkPasses when the passes are initially run.

**NOTE**: Some macros and function names are duplicated between the two classes. Make sure that you modify the correct one!

#### `ClientNetworkManager`
This contains the functions used for network communication for the client.

### `ServerNetworkManager`
Similar to ClientNetworkManager, but for the server.

### Network Passes
This contains the render pass for the network pass logic. Four functions are used, depending on what the pass was initialized with (either client send/receive or server send/receive).

#### `NetworkPass`
Currently unused, replaced by the specific client/server sending/receiving passes.

#### `NetworkClientRecvPass`
A pass that sets off the client receiving thread on the first run. On subsequent runs (i.e. for every frame rendered on the client), it updates the value of `numFramesBehind` which is used by `PredictionPass`.

#### `NetworkClientSendPass`
A pass that sets off the client sending thread on the first run. On subsequent runs, it signals the client sending thread to send the camera data such that the camera data is sent exactly once for every frame rendered on the client.

#### `NetworkServerRecvPass`
A pass that sets off the server receiving thread on the first run. On subsequent runs, it waits for the receiving thread for the camera data to be updated before proceeding with rendering, because there is no point in continuing to render the same scene if the client has not sent over scene changes.

#### `NetworkServerSendPass`
A pass that sets off the server sending thread on the first run. On subsequent runs, it signals the server sending thread to send the visibility data.

### Memory Transfer Passes

#### `MemoryTransferPassClientCPU_GPU`
This render pass simply uploads the texture from the CPU vector to the GPU.

#### `MemoryTransferPassServerGPU_CPU`
This render pass simply downloads the texture from the GPU to the CPU vector.

#### `FrameData`
A struct storing the data for one frame.

#### `NetworkUtils`
These are some utility functions that are used by the networking parts of the program.

#### `PredictionPass`
A pass that offsets the received visibility data at the client to fit the currently rendered frame.

#### `SimulateDelayPass`
A pass that provides an optional delay to the networking parts of the program, by storing data in a buffer temporarily. This may be used for simulating an ideal environment where the network latency results in a delay of an exact number of frames, and this delay is constant. It is useful for testing or for recording demo videos.

#### `VisibilityPass`  
This pass performs ray tracing to each light to generate a texture where each pixel stores a bitmask of whether each light is visible (used on the server).

#### `VShadingPass`
This pass combines the visibility bitmap with the GBuffer information to produce the final shaded image (used on the client).

### Presets
The purpose of the creation of Presets is to allow us to eyeball changes between different rendering pipelines. This is much more convenient than running two compiled versions of the program, or using screenshots, and should increase productivity.

If we have a pipeline like so, where we have multiple options for certain passes:
```
pipeline->setPassOptions(0, {
    JitteredGBufferPass::create(),  // Option 1
    LightProbeGBufferPass::create() // Option 2
});
pipeline->setPass(1, AmbientOcclusionPass::create("Ambient Occlusion"));
pipeline->setPassOptions(2, {
    LambertianPlusShadowPass::create("Lambertian Plus Shadows"), // Option 1
    SimpleDiffuseGIPass::create("Simple Diffuse GI Ray"),        // Option 2
    GGXGlobalIlluminationPass::create("Global Illum., GGX BRDF") // Option 3
});
pipeline->setPass(3, CopyToOutputPass::create());
pipeline->setPass(4, SimpleAccumulationPass::create(ResourceManager::kOutputChannel));
```
We can allow for a quick selection of presets. First, we add the presets:
```
// Presets are "1-indexed", option 0 is the null option to disable the pass
std::vector<uint32_t> lambertianShadingOptions    = { 1, 0, 1, 1, 1 }; // Use Jittered GBuffer and Lambertian pass
std::vector<uint32_t> diffuseGIShadingOptions     = { 2, 0, 2, 1, 1 }; // Use LightProbe GBuffer and diffuse GI pass
std::vector<uint32_t> ggxGIShadingOptions         = { 2, 0, 3, 1, 1 }; // Use LightProbe GBuffer and GGX GI pass
std::vector<uint32_t> justAOOptions               = { 1, 1, 0, 1, 1 }; // Use Jittered GBuffer and AO pass
pipeline->setPresets({
    // PresetData takes the preset's descriptor name (displayed in the dropdown), 
    // the output pass to display, and the selected options
    RenderingPipeline::PresetData("Lambertian Shading", "Lambertian Plus Shadows", lambertianShadingOptions),
    RenderingPipeline::PresetData("Diffuse GI Shading", "Simple Diffuse GI Ray", diffuseGIShadingOptions),
    RenderingPipeline::PresetData("Light Probe GGX GI Shading", "Global Illum., GGX BRDF", ggxGIShadingOptions),
    RenderingPipeline::PresetData("Ambient Occlusion", "Ambient Occlusion", justAOOptions)
});
```
Then, when we run the program and select a preset,  
![Select Preset](images/select_preset.png)  

It will automatically select the passes we intended, as well as make the `CopyToOutputPass` show the desired buffer:  
![Preset selected](images/selected_preset.png)  

### View All GUIs checkbox
This is rather straightforward, simply click the checkbox:  
![Selected checkbox](images/checkbox.png)  
will select all the GUIs:  
![Clicked checkbox](images/checkbox_clicked.png)  

#### Behavior details:
- When any GUIs are disabled, the "view-all" checkbox will automatically uncheck itself
- If you manually set all the GUIs, it will automatically check itself
- Checking it when only some GUIs are open will open all GUIs
- Unchecking it will cause all GUIs to close 

### G-Buffer
The G-Buffer has multiple values compacted into a single texture. We have a utility pass made for previewing it, called `DecodeGBufferPass`. To use it, run the program in debug mode, and select the "Preview GBuffer" mode.  
![Preview GBuffer](images/preview_gbuffer.png)  
Then make sure that the "Decode GBuffer Pass" GUI is enabled, and select the pass you would like to preview.  
![Select GBuffer option](images/select_gbuffer_option.png)  
The methods available to retrieve data from the G-Buffer are available in `DecodeGBufferPass.ps.hlsl`.

## Changes to Falcor
There were several changes to the Falcor base code. Hence, before upgrading the Falcor version to the latest version, you may do so, but understand that these changes must be made.

- Necessary changes
    - Solution file
        - The solution file must be updated to include the `hrender.vcxproj` project file
    - `Falcor/Core/API/Texture.cpp/h`
        - Addition of the `getTextureData()` function to retrieve a texture from the GPU to the CPU
        - Addition of `apiInitPub`, which is just a public version of `apiInit`, which makes the private function available to the user. This is used to upload a texture from the CPU to the GPU
- Unnecessary changes
    - Changes were made to `CopyContext.cpp/h`, `D3D12CopyContext.cpp` and `VKCopyContext.cpp` to add support for preallocated CPU arrays that GPU to CPU transfer would use, but they did not speed up the performance