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
#include "SVGFBitwiseUtils.hlsli"

Texture2D   gMotion;

Texture2D<uint>   gVisTex;
Texture2D<uint>   gAoTex;
Texture2D<uint>   gPrevVisTex;
Texture2D<uint>   gPrevAoTex;
Texture2D<float4>   gPrevMoments;
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


void loadVisAo(int2 fragCoord, out uint vis, out uint ao)
{
    //float3 albedo = gAlbedo[fragCoord].rgb;
    uint v = gVisTex[fragCoord];
    uint a = gAoTex[fragCoord];

    //direct   = gPerformDemodulation ? demodulate(d, albedo) : d;
    //indirect = gPerformDemodulation ? demodulate(i, albedo) : i;
    vis = v;
    ao = a;
}

bool isReprjValid(int2 coord, float Z, float Zprev, float fwidthZ, float3 normal, float3 normalPrev, float fwidthNormal, const int2 imageDim)
{
    // check whether reprojected pixel is inside of the screen
    if (any(coord < int2(1, 1)) || any(coord > imageDim - int2(1, 1)))
        return false;
    // check if deviation of depths is acceptable
    if (abs(Zprev - Z) / (fwidthZ + 1e-2) > 10.0)
        return false;
    // check normals for compatibility
    if (distance(normal, normalPrev) / (fwidthNormal + 1e-2) > 16.0)
        return false;

    return true;
}

bool loadPrevData(float2 fragCoord, out uint prevVisTex, out float prevAoTex, out float4 prevMoments, out float historyLength)
{
    const int2 ipos = fragCoord;
    const float2 imageDim = float2(getTextureDims(gPrevMoments, 0));

    const uint lightCount = 32;
    
    // xy = motion, z = length(fwidth(pos)), w = length(fwidth(normal))
    float4 motion = gMotion[ipos]; 
    float normalFwidth = motion.w;

    // +0.5 to account for texel center offset
    const int2 iposPrev = int2(float2(ipos) + motion.xy * imageDim + float2(0.5,0.5));

    // stores: linearZ, max derivative of linearZ, z_prev, objNorm (.zw are not used here)
    float4 depth = gLinearZ[ipos];
    float3 normal = octToDir(asuint(depth.w));

    prevVisTex = 0x00000000;
    prevAoTex = 0.0;
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

        v[sampleIdx] = isReprjValid(iposPrev, depth.x, depthPrev.x, depth.y, normal, normalPrev, normalFwidth, imageDim);

        valid = valid || v[sampleIdx];
    }    

    float lights[32]; // Accumulates interpolated light values
    
    int lightIndex;
        // Just in case not initialized, can remove if confirmed
    for (lightIndex = 0; lightIndex < lightCount; lightIndex++)
    {
        lights[lightIndex] = 0;
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
                prevAoTex += w[sampleIdx] * gPrevAoTex[loc];
                prevMoments  += w[sampleIdx] * gPrevMoments[loc];
                sumw         += w[sampleIdx];
                
                uint sampleVis = gPrevVisTex[loc];
                for (lightIndex = 0; lightIndex < lightCount; lightIndex++)
                {
                    lights[lightIndex] += (1 & sampleVis);
                    sampleVis = sampleVis >> 1;
                }
            }
        }

        // Bitwise operation on the visibility bitmap
        loadPrevVis2x2(posPrev, v, gPrevVisTex, prevVisTex);
        
        // redistribute weights in case not all taps were used
        valid = (sumw >= 0.01);
        prevVisTex = 0x0;
        prevAoTex = valid ? prevAoTex / sumw : 0.0;
        prevMoments  = valid ? prevMoments / sumw  : float4(0, 0, 0, 0);
        
        if (valid)
        {
            for (lightIndex = 0; lightIndex < lightCount; lightIndex++)
            {
                int bit = lights[lightIndex] / sumw > 0.5 ? 1 : 0;
                bit = bit << lightIndex;
                prevVisTex += bit;
            }
        }  
    }
    
    if(!valid) // perform cross-bilateral filter in the hope to find some suitable samples somewhere
    {
        float cnt = 0.0;

        // this code performs a binary decision for each tap of the cross-bilateral filter
        const int radius = 1;
        for (int yy = -radius; yy <= radius; yy++)
        {
            for (int xx = -radius; xx <= radius; xx++)
            {
                int2 p = iposPrev + int2(xx, yy);
                float4 depthFilter = gPrevLinearZ[p];
                float3 normalFilter = octToDir(asuint(depthFilter.w));

                if (isReprjValid(iposPrev, depth.x, depthFilter.x, depth.y, normal, normalFilter, normalFwidth, imageDim))
                {
                    prevAoTex += float(gPrevAoTex[p]);
                    prevMoments += gPrevMoments[p];
                    cnt += 1.0;
                    
                    uint sampleVis = gPrevVisTex[p];
                    for (lightIndex = 0; lightIndex < lightCount; lightIndex++)
                    {
                        lights[lightIndex] += (1 & sampleVis);
                        sampleVis = sampleVis >> 1;
                    }
                }
            }
        }
        
        float4 depthFilter = gPrevLinearZ[p];
        float3 normalFilter = octToDir(asuint(depthFilter.w));
        // Bitwise operation on the visibility bitmap
        loadPrevVis3x3(iposPrev, depth, depthFilter.x, 
                        normal, normalFilter, normalFwidth,
                        gPrevVisTex, prevVisTex);
        
        if (cnt > 0)
        {
            valid = true;
            prevAoTex /= cnt;
            prevMoments  /= cnt;
            for (lightIndex = 0; lightIndex < lightCount; lightIndex++)
            {
                int bit = lights[lightIndex] > cnt / 2 ? 1 : 0;
                bit = bit << lightIndex;
                prevVisTex += bit;
            }
        }

    }

    if (valid)
    {
        // crude, fixme
        historyLength = gHistoryLength.Load(int3(iposPrev, 0)).r;
    }
    else
    {
        prevVisTex = 0x00000000;
        prevAoTex = 0.0;
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
    uint OutVisTex        : SV_TARGET0;
    uint OutAoTex         : SV_TARGET1;
    float4 OutMoments       : SV_TARGET2;
    float OutHistoryLength  : SV_TARGET3;
};

PS_OUT main(FullScreenPassVsOut vsOut)
{
    float4 fragCoord = vsOut.posH;

    const int2 ipos = fragCoord.xy;
    uint vis, ao;
    loadVisAo(ipos, vis, ao);
    float historyLength;
    uint prevVisTex;
    float prevAoTex;
    float4 prevMoments;
    bool success = loadPrevData(fragCoord.xy, prevVisTex, prevAoTex, prevMoments, historyLength);
    historyLength = min( 32.0f, success ? historyLength + 1.0f : 1.0f );

    // this adjusts the alpha for the case where insufficient history is available.
    // It boosts the temporal accumulation to give the samples equal weights in
    // the beginning.
    const float alpha        = success ? max(gAlpha,        1.0 / historyLength) : 1.0;
    const float alphaMoments = success ? max(gMomentsAlpha, 1.0 / historyLength) : 1.0;

    // compute first two moments of luminance
    float4 moments;
    // r = ao, b = 2nd moment of ao, g = variance of visTex, a = variance of aoTex
    moments.r = ao;
    moments.b = moments.r * moments.r;
    moments.g = 0;
    moments.a = 0;

    // temporal integration of the ao moments
    moments.rb = lerp(prevMoments.rb, moments.rb, alphaMoments);

    PS_OUT psOut;

    // temporal integration of visTex and AO
    psOut.OutVisTex = prevVisTex; // With alpha < 0.5 (=0.2), lerp eqn = (prev * (1-alpha) + curr * alpha), only the bits in prevVisTex matter when doing bitwise lerp
    psOut.OutAoTex = uint(lerp(prevAoTex, float(ao), alpha));

    // Variance of visibility is just the XOR of current vs previous bitmaps
    float2 variance = max(float2(0, 0), float2(asfloat(prevVisTex ^ vis), moments.b - moments.r * moments.r));

    // variance is propagated using moments.ga
    moments.ga = variance;
    
    psOut.OutMoments = moments;
    psOut.OutHistoryLength = historyLength;

    return psOut;
}
