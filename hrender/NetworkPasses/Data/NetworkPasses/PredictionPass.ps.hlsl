
// Input data that needs to be set by the C++ code
cbuffer PredictionCb
{
    // float3 gCamMotionVec;
    float4x4 gOldViewProjMat;
    float gCamNearZ;
    float gCamFarZ;
    int gTexHeight;
    int gTexWidth;
}

// Input texture that needs to be set by the C++ code
Texture2D<float4> gWorldPos;

// Output texture of motion vectors
struct PsOut
{
    float4 motionVec : SV_Target0;
};

// NaN constant
const static float NaN = 0.0f / 0.0f;

PsOut main(float2 texC : TEXCOORD, float4 screenSpacePos : SV_Position)
{
    // // Get pixel position in screen coordinates
	// float2 pixelPos = screenSpacePos.xy;

	// // Get the depth buffer value at this pixel
	// float depth = screenSpacePos.z;

    // // Using camera data, calculate the actual z-coordinate of the pixel in camera space
    // // The full formula is
    // //             f - n                  f + n
    // //  x = 1 / ( ------- * ( 2y - 1 - ( ------- )))
    // //             -2fn                   f - n
    // // where x is the original z-coordinate in camera space,
    // // and y is the z-buffer/depth-buffer value
    // // (calculated from the matrix used for perspective projection)
    // float diffFarNear = gCamFarZ - gCamNearZ;
    // float sumFarNear = gCamFarZ + gCamNearZ;
    // float pdtFarNear = gCamFarZ * gCamNearZ
    // // TODO: Constants can be calculated in CPU and passed as shader variables
    // //  x = (my + c) ^ -1
    // // where
    // //           f - n                      f - n         f + n 
    // //  m = 2 * -------     and     c = -( ------- )(1 + -------)
    // //           -2fn                       -2fn          f - n 

    // // Scale depth buffer value from [0, 1] to [-1, 1]
    // float scaledDepth = 2 * depth - 1;
    // // Calculate actual z-coord value
    // float pixelZValue = 1 / ((diffFarNear / (-2 * pdtFarNear)) * (scaledDepth - (sumFarNear / diffFarNear)));


    // Get point in world space from input texture
    float4 worldPos = gWorldPos[screenSpacePos.xy];

	// Calculate the old position of this point in NDC space
    // in the old frame that we have the visibility buffer for
    float4 oldClipSpacePos = mul(worldPos, gOldViewProjMat);

    // Construct the struct to be returned by the shader
    PsOut motionVecBufOut;

    // Do clipping
    if (oldClipSpacePos.w == 0)
    {
        motionVecBufOut.motionVec.xy = float2(0, 0);
    }
    else
    {
        // Do perspective division
        float4 oldNdcSpacePos = oldClipSpacePos / oldClipSpacePos.w;

        // If screen space is out-of-bounds, we ignore it and
        // set the motion vector to (NaN, NaN) to reuse the
        // same visibility data for the current fragment
        if (oldNdcSpacePos.x < -1 || oldNdcSpacePos.x > 1 ||
            oldNdcSpacePos.y < -1 || oldNdcSpacePos.y > 1 ||
            oldNdcSpacePos.z < -1 || oldNdcSpacePos.z > 1)
        {
            motionVecBufOut.motionVec.xy = float2(NaN, NaN);
        }
        else
        {
            // Calculate old screen space position
            float2 oldScreenSpacePos = float2((oldNdcSpacePos.x + 1) * 0.5 * gTexWidth,
                                              (-oldNdcSpacePos.y + 1) * 0.5 * gTexHeight);

            // Calculate the motion vector by taking the difference in
            // the screen space positions for the same world space
            // point as seen by the camera in different frames
            //
            // Motion vector points from old point to current point
            motionVecBufOut.motionVec.xy = screenSpacePos.xy - oldScreenSpacePos;

            // In OffsetBuffer.ps.hlsl, we just subtract the motion vector
            // from the fragment screen space position to get the coordinate
            // that we use to read from the received visibility buffer.
            // 
            // This gives us the predicted visibility that better matches the
            // scene as seen from the new camera position in the current frame.
        }
    }

    // Set the other values for the float4
    motionVecBufOut.motionVec.zw = float2(0, 1);

    // Return the motion vector
    return motionVecBufOut;
}