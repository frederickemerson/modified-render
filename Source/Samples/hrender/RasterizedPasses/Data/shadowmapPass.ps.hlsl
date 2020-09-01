#include "shadowmapCommon.hlsli"

cbuffer ShadowPsVars
{
    // Distance of far plane squared
    float  gShadowFar2;
    // Position of the light we are rendering the view from
    float3 gLightPosW;
};

float main(ShadowMapVertexOut vsOut, uint primID : SV_PrimitiveID) : SV_Target0
{
    float3 viewVec = vsOut.posW - gLightPosW;
    // Return the normalized distance^2 in range [0, 1]. We use ^2 to prevent using sqrt.
    return dot(viewVec, viewVec) / gShadowFar2;
}
