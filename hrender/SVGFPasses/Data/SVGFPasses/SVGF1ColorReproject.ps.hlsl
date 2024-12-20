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
#include "SVGFPackNormal.hlsli"
#include "SVGFEdgeStoppingFunctions.hlsli"

Texture2D   gMotion;

Texture2D   gColor;
Texture2D   gPrevColor;
Texture2D   gPrevMoments;
//Texture2D   gAlbedo;

Texture2D   gLinearZ;
Texture2D   gPrevLinearZ;

Texture2D   gHistoryLength;

cbuffer PerImageCB : register(b0)
{
    float       gAlpha;
    float       gMomentsAlpha;
    //bool        gPerformDemodulation;
};

// Given color x and albedo, retrieve the illumination
float3 demodulate(float3 x, float3 albedo)
{
    return x / max(albedo, float3(0.001, 0.001, 0.001));
}

void loadColor(int2 fragCoord, out float3 color)
{
    //float3 albedo = gAlbedo[fragCoord].rgb;

    //direct   = gPerformDemodulation ? demodulate(d, albedo) : d;
    //indirect = gPerformDemodulation ? demodulate(i, albedo) : i;
    color = gColor[fragCoord].rgb;
}

bool isReprjValid(int2 coord, float Z, float Zprev, float fwidthZ, float3 normal, float3 normalPrev, float fwidthNormal)
{
    const int2 imageDim = getTextureDims(gColor, 0);
    // check whether reprojected pixel is inside of the screen
    if(any(coord < int2(1,1)) || any(coord > imageDim - int2(1,1))) return false;
    // check if deviation of depths is acceptable
    if(abs(Zprev - Z) / (fwidthZ + 1e-2) > 10.0) return false;
    // check normals for compatibility
    if(distance(normal, normalPrev) / (fwidthNormal + 1e-2) > 16.0) return false;

    return true;
}

bool loadPrevData(float2 fragCoord, out float4 prevColor, out float4 prevMoments, out float historyLength)
{
    const int2 ipos = fragCoord;
    const float2 imageDim = float2(getTextureDims(gColor, 0));

    // xy = motion, z = length(fwidth(pos)), w = length(fwidth(normal))
    float4 motion = gMotion[ipos]; 
    float normalFwidth = motion.w;

    // +0.5 to account for texel center offset
    const int2 iposPrev = int2(float2(ipos) + motion.xy * imageDim + float2(0.5,0.5));

    // stores: linearZ, max derivative of linearZ, z_prev, objNorm (.zw are not used here)
    float4 depth = gLinearZ[ipos];
    float3 normal = octToDir(asuint(depth.w));

    prevColor   = float4(0,0,0,0);
    prevMoments  = float4(0,0,0,0);

    bool v[4];
    const float2 posPrev = floor(fragCoord.xy) + motion.xy * imageDim;
    int2 offset[4] = { int2(0, 0), int2(1, 0), int2(0, 1), int2(1, 1) };
    
    // check for all 4 taps of the bilinear filter for validity
    bool valid = false;
    for (int sampleIdx = 0; sampleIdx < 4; sampleIdx++)
    { 
        int2 loc = int2(posPrev) + offset[sampleIdx];
        float4 depthPrev = gPrevLinearZ[loc];
        float3 normalPrev = octToDir(asuint(depthPrev.w));

        v[sampleIdx] = isReprjValid(iposPrev, depth.x, depthPrev.x, depth.y, normal, normalPrev, normalFwidth);

        valid = valid || v[sampleIdx];
    }    

    if (valid) 
    {
        float sumw = 0;
        float x = frac(posPrev.x);
        float y = frac(posPrev.y);

        // bilinear weights
        float w[4] = { (1 - x) * (1 - y), 
                            x  * (1 - y), 
                       (1 - x) *      y,
                            x  *      y };

        // perform the actual bilinear interpolation
        for (int sampleIdx = 0; sampleIdx < 4; sampleIdx++)
        {
            int2 loc = int2(posPrev) + offset[sampleIdx];            
            if (v[sampleIdx])
            {
                prevColor   += w[sampleIdx] * gPrevColor[loc];
                prevMoments  += w[sampleIdx] * gPrevMoments[loc];
                sumw         += w[sampleIdx];
            }
        }

        // redistribute weights in case not all taps were used
        valid = (sumw >= 0.01);
        prevColor = valid ? prevColor / sumw : float4(0, 0, 0, 0);
        prevMoments  = valid ? prevMoments / sumw  : float4(0, 0, 0, 0);
    }
    if(!valid) // perform cross-bilateral filter in the hope to find some suitable samples somewhere
    {
        float cnt = 0.0;

        // this code performs a binary descision for each tap of the cross-bilateral filter
        const int radius = 1;
        for (int yy = -radius; yy <= radius; yy++)
        {
            for (int xx = -radius; xx <= radius; xx++)
            {
                int2 p = iposPrev + int2(xx, yy);
                float4 depthFilter = gPrevLinearZ[p];
                float3 normalFilter = octToDir(asuint(depthFilter.w));

                if ( isReprjValid(iposPrev, depth.x, depthFilter.x, depth.y, normal, normalFilter, normalFwidth) )
                {
                    prevColor += gPrevColor[p];
                    prevMoments += gPrevMoments[p];
                    cnt += 1.0;
                }
            }
        }
        if (cnt > 0)
        {
            valid = true;
            prevColor /= cnt;
            prevMoments  /= cnt;
        }

    }

    if (valid)
    {
        // crude, fixme
        historyLength = gHistoryLength.Load(int3(iposPrev, 0)).r;
    }
    else
    {
        prevColor = float4(0, 0, 0, 0);
        prevMoments = float4(0,0,0,0);
        historyLength = 0;
    }

    return valid;
}

// not used currently
float computeVarianceScale(float numSamples, float loopLength, float alpha)
{
    const float aa = (1.0 - alpha) * (1.0 - alpha);
    return (1.0 - pow(aa, min(loopLength, numSamples))) / (1.0 - aa);
}

struct PS_OUT
{
    float4 OutColor        : SV_TARGET0;
    float4 OutMoments       : SV_TARGET1;
    float OutHistoryLength  : SV_TARGET2;
};

PS_OUT main(FullScreenPassVsOut vsOut)
{
    float4 fragCoord = vsOut.posH;

    const int2 ipos = fragCoord.xy;
    float3 color;
    loadColor(ipos, color);
    float historyLength;
    float4 prevColor, prevMoments;
    bool success = loadPrevData(fragCoord.xy, prevColor, prevMoments, historyLength);
    historyLength = min( 32.0f, success ? historyLength + 1.0f : 1.0f );

    // this adjusts the alpha for the case where insufficient history is available.
    // It boosts the temporal accumulation to give the samples equal weights in
    // the beginning.
    const float alpha        = success ? max(gAlpha,        1.0 / historyLength) : 1.0;
    const float alphaMoments = success ? max(gMomentsAlpha, 1.0 / historyLength) : 1.0;

    // compute first two moments of luminance
    float4 moments;
    // We only calculate one color at a time, so b and a are unused
    moments.r = luminance(color);
    moments.b = 0.0;
    moments.g = moments.r * moments.r;
    moments.a = 0.0;

    // temporal integration of the moments
    moments = lerp(prevMoments, moments, alphaMoments);

    PS_OUT psOut;

    psOut.OutMoments = moments;
    psOut.OutHistoryLength = historyLength;

    float2 variance = max(float2(0,0), moments.ga - moments.rb * moments.rb);

    //variance *= computeVarianceScale(16, 16, alpha);

    // temporal integration of direct and indirect illumination
    psOut.OutColor  = lerp(prevColor,   float4(color,   0), alpha);

    // variance is propagated through the alpha channel
    psOut.OutColor.a = variance.r;

    return psOut;
}
