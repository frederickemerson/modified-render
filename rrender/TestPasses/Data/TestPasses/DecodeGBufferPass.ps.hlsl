#include "../../../DxrTutorCommonPasses/Data/CommonPasses/packingUtils.hlsli"
#include "../../GBufferComponents.slang"

// Textures from the GBuffer pass
Texture2D<float4> gWsPos;
Texture2D<float4> gWsNorm;
Texture2D<float4> gTexData;

cbuffer DecoderCB
{
    // The buffer to render
    uint gBufComponent; // Refer to GBufferComponents.slang
};

float4 main(float2 texC : TEXCOORD, float4 pos : SV_Position) : SV_Target0
{
    int2 ipos        = int2(pos.xy);

    float4 result;
    switch (gBufComponent)
    {
    case GBufComponent::WorldPosition:
        result = gWsPos[ipos];
        break;
    case GBufComponent::WorldNormal:
        result = gWsNorm[ipos];
        break;
    case GBufComponent::MatlDiffuse:
        result = unpackUnorm4x8(asuint(gTexData[ipos].x));
        break;
    case GBufComponent::MatlSpecular:
        result = unpackUnorm4x8(asuint(gTexData[ipos].y));
        break;
    case GBufComponent::MatlEmissive:
        result = float4(unpackUnorm4x8(asuint(gTexData[ipos].z)).rgb, 1.0f);
        break;
    case GBufComponent::MatlDoubleSided:
        float doubleSided = unpackUnorm4x8(asuint(gTexData[ipos].z)).a;
        result = float4(doubleSided, doubleSided, doubleSided, 1.0f);
        break;
    case GBufComponent::MatlExtra:
        result = unpackUnorm4x8(asuint(gTexData[ipos].w));
        break;
    default:
        result = float4(0.f, 0.f, 0.f, 1.0f);
        break;
    }
    return result;
}
