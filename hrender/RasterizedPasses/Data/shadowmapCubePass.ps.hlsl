#include "shadowmapCommon.hlsli"

cbuffer ShadowPsVars
{
    // Distance of far plane squared
    float  gShadowFar2;
    // Position of the light we are rendering the view from
    float3 gLightPosW;
    // Which face index are we writing to
    float  gFace;
    // Resolution of the cube shadow map used
    uint   gCubeShadowMapRes;
};

RWTexture2D<float> gCubeMap;

float main(ShadowMapVertexOut vsOut, uint primID : SV_PrimitiveID) : SV_Target0

{
    // Point light shadow mapping is a bit more involved, we need to compute the distance
    // in world space, since we won't be using the light's view matrix to retrieve the shadow
    // map entry, but a directional vector.
    float3 viewVec = vsOut.posW - gLightPosW;
    // Return the normalized distance^2 in range [0, 1]. We use ^2 to prevent using sqrt.
    return dot(viewVec, viewVec) / gShadowFar2;
}
