
// Input data that needs to be set by the C++ code
cbuffer OffsetCb
{
    int gTexHeight;
    int gTexWidth;
    float gOffsetFactor;
    int gUnknownFragMode;
}

// Input textures that needs to be set by the C++ code
Texture2D<uint> gVisBufferOrig;
Texture2D<uint> gAOBufferOrig;
Texture2D<float4> gDirectIllumOrig;
Texture2D<float4> gIndirectIllumOrig;
Texture2D<float4> gMotionVectors;

// Output texture with offset visibility buffer
struct PsOut
{
	//uint visOffset : SV_Target0;
    //uint AoOffset  : SV_Target1;
    float4 directIllumOffset : SV_Target0;
    float4 indirectIllumOffset  : SV_Target1;
};

PsOut main(float2 texC : TEXCOORD, float4 currScreenSpacePos : SV_Position)
{
    // Construct the struct to be returned by the shader
    PsOut motionVecBufOut;

    // Screen space position as a 2D vector
    float2 screenCoords = currScreenSpacePos.xy;

    // Retrieve raw motion vector
    float2 rawMotionVec = gMotionVectors[screenCoords].xy;

    // Check for unknown fragment marked by NaN
    if (isnan(rawMotionVec.x) && isnan(rawMotionVec.y))
    {
        if (gUnknownFragMode == 0)
        {
            // Copy mode
            motionVecBufOut.directIllumOffset = gDirectIllumOrig[screenCoords];
            motionVecBufOut.indirectIllumOffset = gIndirectIllumOrig[screenCoords];
        }
        else if (gUnknownFragMode == 1)
        {
            // Fill with shadow mode
            motionVecBufOut.directIllumOffset = float4(float3(0.0), 1.0);
            motionVecBufOut.indirectIllumOffset = float4(float3(0.0), 1.0);
        }
        else // (gUnknownFragMode == 2)
        {
            // Fill with light mode
            motionVecBufOut.directIllumOffset = float4(float3(1.0), 1.0);
            motionVecBufOut.indirectIllumOffset = float4(float3(1.0), 1.0);
        }
        return motionVecBufOut;
    }

    // Multiply by offsetFactor
    float2 motionVec = rawMotionVec * gOffsetFactor;

    // Offset the screen space coordinates by the motion vector
    float2 offsetCoords = screenCoords - motionVec;

    // Redundant boundary check, just in case (should be already done in PredictionPass)
    if (offsetCoords.x < 0 || offsetCoords.x > gTexWidth ||
        offsetCoords.y < 0 || offsetCoords.y > gTexHeight)
    {
        offsetCoords = screenCoords;
    }

    // Read original illumination using the offset coordinates
    float4 offsetDirectIllumInfo = gDirectIllumOrig[offsetCoords];
    float4 offsetIndirectIllumInfo = gIndirectIllumOrig[offsetCoords];

    // Return the offset visibility buffer
    motionVecBufOut.directIllumOffset = offsetDirectIllumInfo;
    motionVecBufOut.indirectIllumOffset = offsetIndirectIllumInfo;
    return motionVecBufOut;
}