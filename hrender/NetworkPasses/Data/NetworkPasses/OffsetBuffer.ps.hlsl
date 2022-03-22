
// Input data that needs to be set by the C++ code
cbuffer OffsetCb
{
    int gTexHeight;
    int gTexWidth;
    float gOffsetFactor;
}

// Input textures that needs to be set by the C++ code
Texture2D<float4> gVisBufferOrig;
Texture2D<float4> gMotionVectors;

// Output texture with offset visibility buffer
struct PsOut
{
    float4 visOffset : SV_Target0;
};

PsOut main(float2 texC : TEXCOORD, float4 currScreenSpacePos : SV_Position)
{
    // Construct the struct to be returned by the shader
    PsOut motionVecBufOut;

    // Screen space position as a 2D vector
    float2 screenCoords = currScreenSpacePos.xy;

    // Retrieve motion vector and multiply by offsetFactor
    float2 motionVec = gMotionVectors[screenCoords].xy * gOffsetFactor;

    // Offset the screen space coordinates by the motion vector
    float2 offsetCoords = screenCoords - motionVec;

    // Redundant boundary check, just in case (should be already done in PredictionPass)
    if (offsetCoords.x < 0 || offsetCoords.x > gTexWidth ||
        offsetCoords.y < 0 || offsetCoords.y > gTexHeight)
    {
        offsetCoords = screenCoords;
    }

    // Read original visibility buffer using the offset coordinates
    float4 offsetVisibilityInfo = gVisBufferOrig[offsetCoords];

    // Return the offset visibility buffer
    motionVecBufOut.visOffset = offsetVisibilityInfo;
    return motionVecBufOut;
}