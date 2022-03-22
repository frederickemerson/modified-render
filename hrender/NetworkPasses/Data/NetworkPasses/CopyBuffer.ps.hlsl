
// Input texture that needs to be set by the C++ code
Texture2D<float4> gVisBufferOrig;

// Output texture with offset visibility buffer
struct PsOut
{
    float4 visCopy : SV_Target0;
};

PsOut main(float2 texC : TEXCOORD, float4 screenSpacePos : SV_Position)
{
    // Construct the struct to be returned by the shader
    PsOut motionVecBufOut;

    // Return the same visibility buffer
    motionVecBufOut.visCopy = gVisBufferOrig[screenSpacePos.xy];
    return motionVecBufOut;
}