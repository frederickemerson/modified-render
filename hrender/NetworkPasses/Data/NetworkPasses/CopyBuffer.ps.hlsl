
// Input texture that needs to be set by the C++ code
Texture2D<float4> gVisBufferOrig;
Texture2D<float4> gAOBufferOrig;

// Output texture with offset visibility buffer
struct PsOut
{
    float4 visCopy : SV_Target0;
    float4 AOCopy  : SV_Target1;
};

PsOut main(float2 texC : TEXCOORD, float4 screenSpacePos : SV_Position)
{
    // Construct the struct to be returned by the shader
    PsOut motionVecBufOut;

    // Return the same buffers
    motionVecBufOut.visCopy = gVisBufferOrig[screenSpacePos.xy];
    motionVecBufOut.AOCopy = gAOBufferOrig[screenSpacePos.xy];
    return motionVecBufOut;
}