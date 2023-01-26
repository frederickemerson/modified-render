/**********************************************************************************************************************
# Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without modification, are permitted provided that the
# following conditions are met:
#  * Redistributions of code must retain the copyright notice, this list of conditions and the following disclaimer.
#  * Neither the name of NVIDIA CORPORATION nor the names of its contributors may be used to endorse or promote products
#    derived from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.  IN NO EVENT
# SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA,
# OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
# LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
# ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
**********************************************************************************************************************/

import Utils.Color.ColorHelpers; // Contains function for computing luminance
import Experimental.Scene.Lights.LightHelpers;

#include "SVGFCommon.hlsli"
#include "SVGFEdgeStoppingFunctions.hlsli"
#include "SVGFPackNormal.hlsli"

Texture2D<uint>   gVisTex;
Texture2D<uint>   gAoTex;
Texture2D   gMoments;
Texture2D   gHistoryLength;
Texture2D   gCompactNormDepth;

cbuffer PerImageCB : register(b0)
{
    float       gPhiColor;
    float       gPhiNormal;
};

struct PS_OUT
{
    uint OutVisTex   : SV_TARGET0;
    uint OutAoTex : SV_TARGET1;
    float2 OutVar : SV_TARGET2;
};

PS_OUT main(FullScreenPassVsOut vsOut)
{

    float4 fragCoord = vsOut.posH;
    int2 ipos = int2(fragCoord.xy);

    float h = gHistoryLength[ipos].r;
    int2 screenSize = getTextureDims(gHistoryLength, 0);

    if (h < 4.0) // not enough temporal history available
    {
        uint sumVis        = 0x00000000;
        uint sumAo         = 0x00000000;
        float sumW = 0.0;
        float4  sumMoments = float4(0.0, 0.0, 0.0, 0.0);

        const uint visCenter = gVisTex[ipos];
        const uint aoCenter = gAoTex[ipos];

        float3 normalCenter;
        float2 zCenter;
        fetchNormalAndLinearZ(gCompactNormDepth, ipos, normalCenter, zCenter);

        PS_OUT psOut;

        if (zCenter.x < 0)
        {
            // current pixel does not a valid depth => must be envmap => do nothing
            psOut.OutVisTex   = visCenter;
            psOut.OutAoTex = aoCenter;
            return psOut;
        }

        const float phiDepth     = max(zCenter.y, 1e-8) * 3.0;

        // compute first and second moment spatially. This code also applies cross-bilateral
        // filtering on the input samples
        const int radius = 3;

        float lights[32]; // Accumulates interpolated light values
        const uint lightCount = gScene.getLightCount();
        
        int lightIndex;
        // Just in case not initialized, can remove if confirmed
        for (lightIndex = 0; lightIndex < lightCount; lightIndex++)
        {
            lights[lightIndex] = 0;
        }
        
        for (int yy = -radius; yy <= radius; yy++)
        {
            for (int xx = -radius; xx <= radius; xx++)
            {
                const int2 p     = ipos + int2(xx, yy);
                const bool inside = all(p >= int2(0,0)) && all(p < screenSize);
                const bool samePixel = (xx==0) && (yy==0);
                const float kernel = 1.0;

                if (inside)
                {

                    const uint visP  = gVisTex[p];
                    const uint aoP   = gAoTex[p];
                    const float4 momentsP    = gMoments[p];

                    float3 normalP;
                    float2 zP;
                    fetchNormalAndLinearZ(gCompactNormDepth, p, normalP, zP);

                    // Vis and Ao share weights
                    const float w = computeWeightNoLuminance(
                        zCenter.x, zP.x, phiDepth * length(float2(xx, yy)),
                        normalCenter, normalP, gPhiNormal);

                    sumW += w;
                    sumAo += aoP * w;

                    // We do interpolation per bit
                    for (lightIndex = 0; lightIndex < lightCount; lightIndex++)
                    {
                        lights[lightIndex] += (1 & visP) * w;
                        visP = visP >> 1;
                    }
                    
                    sumMoments += momentsP * float4(w.xx, float2(1.0));
                }
            }
        }

        // Clamp sums to >0 to avoid NaNs.
        sumW = max(sumW, 1e-6f);
        sumAo /= sumW;
        
        // We do interpolation per bit
        for (lightIndex = 0; lightIndex < lightCount; lightIndex++)
        {
            int bit = lights[lightIndex] /= sumW > 0.5 ? 1 : 0;
            bit = bit << lightIndex;
            sumVis += bit;
        }

        sumMoments /= float4(sumW.xx, float2(1.0));

        // compute variance for direct and indirect illumination using first and second moments

        float2 variance = max(uint2(0, 0), float2(asfloat(sumVis ^ visCenter), sumMoments.b - sumMoments.r * sumMoments.r));
        
        // give the variance a boost for the first frames
        variance.y *= 4.0 / h;

        // TODO: we might just use the variance and calculate OR between the sample and center (then skip for further)
        psOut.OutVisTex = sumVis;
        psOut.OutAoTex = sumAo;
        psOut.OutVar = variance;
        
        return psOut;
    }
    else
    {
        // do nothing, pass data unmodified
        PS_OUT psOut;

        psOut.OutVisTex = gVisTex[ipos];
        psOut.OutAoTex = gAoTex[ipos];
        psOut.OutVar = gMoments[ipos].zw;
        
        return psOut;
    }
}
