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

#include "SVGFCommon.hlsli"
#include "SVGFEdgeStoppingFunctions.hlsli"
#include "SVGFPackNormal.hlsli"

Texture2D   gColor;
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
    float4 OutColor   : SV_TARGET0;
};

PS_OUT main(FullScreenPassVsOut vsOut)
{

    float4 fragCoord = vsOut.posH;
    int2 ipos = int2(fragCoord.xy);

    float h = gHistoryLength[ipos].r;
    int2 screenSize = getTextureDims(gHistoryLength, 0);

    if (h < 4.0) // not enough temporal history available
    {
        float sumWColor   = 0.0;
        float3 sumColor    = float3(0.0, 0.0, 0.0);
        float4 sumMoments   = float4(0.0, 0.0, 0.0, 0.0);

        const float4 colorCenter    = gColor[ipos];
        const float lColorCenter = luminance(colorCenter.rgb);

        float3 normalCenter;
        float2 zCenter;
        fetchNormalAndLinearZ(gCompactNormDepth, ipos, normalCenter, zCenter);

        PS_OUT psOut;

        if (zCenter.x < 0)
        {
            // current pixel does not a valid depth => must be envmap => do nothing
            psOut.OutColor = colorCenter;
            return psOut;
        }

        const float phiLColor   = gPhiColor;
        const float phiDepth     = max(zCenter.y, 1e-8) * 3.0;

        // compute first and second moment spatially. This code also applies cross-bilateral
        // filtering on the input color samples
        const int radius = 3;

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

                    const float3 colorP     = gColor[p].rgb;
                    const float4 momentsP    = gMoments[p];

                    const float lColorP = luminance(colorP.rgb);

                    float3 normalP;
                    float2 zP;
                    fetchNormalAndLinearZ(gCompactNormDepth, p, normalP, zP);

                    const float w = computeWeight(
                        zCenter.x, zP.x, phiDepth * length(float2(xx, yy)),
                        normalCenter, normalP, gPhiNormal, 
                        lColorCenter, lColorP, phiLColor);

                    sumWColor  += w;
                    sumColor   += colorP * w;

                    sumMoments += momentsP * float4(w, w, 0.0, 0.0);
                }
            }
        }

        // Clamp sums to >0 to avoid NaNs.
        sumWColor = max(sumWColor, 1e-6f);
        

        sumColor   /= sumWColor;
        sumMoments  /= float4(sumWColor.xx, 1.0, 1.0);

        // compute variance for direct and indirect illumination using first and second moments
        float2 variance = sumMoments.ga - sumMoments.rb * sumMoments.rb;

        // give the variance a boost for the first frames
        variance *= 4.0 / h;

        psOut.OutColor = float4(sumColor, variance.r);
        
        return psOut;
    }
    else
    {
        // do nothing, pass data unmodified
        PS_OUT psOut;

        psOut.OutColor = gColor[ipos];

        return psOut;
    }
}
