#include "shadowmapCommon.hlsli"

float main(ShadowMapVertexOut vsOut, uint primID : SV_PrimitiveID) : SV_Target0
{
    // For directional light shadow mapping, the z value is enough
    return vsOut.posH.z / vsOut.posH.w;
}
