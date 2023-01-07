// Include and import common Falcor utilities and data structures
import Scene.Raytracing;
import Scene.Shading;                          // Shading functions, etc   
import Experimental.Scene.Lights.LightHelpers; // Light structures for our current scene

// To avoid two very similar shaders, both textures are RWTexture2D so they can function as input and output.
RWTexture2D<float3> gCompact;
RWTexture2D<float4> gUncompact;

// True if converting from RGBA32Float to R11G11B10 float, false if the other way
cbuffer PerImageCB : register(b0)
{
    bool gIsCompacting;
};

// Values used for RGB -> YUV conversion. Alpha value ignored.
//static float3x3 yuvToRgb = {   1.0,    1.0,   1.0,
//                               0.0, -0.344, 1.772,
//                             1.402, -0.714,   0.0};
//static uint3 offset = { 0.0, 128.0, 128.0 };

[shader("raygeneration")]
void ColorsRayGen()
{
    // Where is this ray on screen?
    uint2 launchIndex = DispatchRaysIndex().xy;
    uint2 launchDim = DispatchRaysDimensions().xy;

    //uint y = (gInput[launchIndex] >> 24) & 0xFF;
    //uint u = (gInput[launchIndex] >> 16) & 0xFF;
    //uint v = (gInput[launchIndex] >> 8) & 0xFF;
    //float3 yuv = { (float) y, (float) u, (float) v };
    //yuv -= offset;
    //float3 rgbOutput = mul(yuv, yuvToRgb) / 255.0;

    //gOutput[launchIndex] = float4(clamp(rgbOutput, float3(0.0, 0.0, 0.0), float3(1.0, 1.0, 1.0)), 0.0);
    
    if (gIsCompacting)
    {
        gCompact[launchIndex] = gUncompact[launchIndex].rgb;
    }
    else
    {
        gUncompact[launchIndex] = float4(gCompact[launchIndex], 1.0f);
    }
    
}
