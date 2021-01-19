import Scene.Raster;
#include "shadowmapCommon.hlsli"

cbuffer ShadowMapBuf
{
    // View-Projection matrix for the point of view of the light
    float4x4 gLightViewProj;
};

ShadowMapVertexOut main(VSIn vIn)
{
    ShadowMapVertexOut vOut;
    float4x4 worldMat = gScene.getWorldMatrix(vIn.meshInstanceID);
    float4 posW = mul(float4(vIn.pos, 1.f), worldMat);
    vOut.posW = posW.xyz;
    vOut.posH = mul(posW, gLightViewProj);
    
    return vOut;
}
