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

#ifndef SVGF_BITWISE_H
#define SVGF_BITWISE_H

import Experimental.Scene.Lights.LightHelpers; // Light structures for our current scene

bool isReprjValid(int2 coord, float Z, float Zprev, float fwidthZ, float3 normal, float3 normalPrev, float fwidthNormal)
{
    const int2 imageDim = getTextureDims(gVisTex, 0);
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

void loadPrevVis2x2(float2 posPrev, bool v[], Texture2D gPrevVisTex, inout uint prevVisTex)
{
    int2 offset[4] = { int2(0, 0), int2(1, 0), int2(0, 1), int2(1, 1) };
    
    float sumw = 0;
    float x = frac(posPrev.x);
    float y = frac(posPrev.y);
    // bilinear weights
    float w[4] = { (1 - x) * (1 - y), x * (1 - y), (1 - x) * y, x * y };
    const uint lightCount = gScene.getLightCount();
    float lights[32]; // Accumulates interpolated light values
    
    int lightIndex;
    // Just in case not initialized, can remove if confirmed
    for (lightIndex = 0; lightIndex < lightCount; lightIndex++)
    {
        lights[lightIndex] = 0;
    }
    
    // We do interpolation per bit
    for (int sampleIdx = 0; sampleIdx < 4; sampleIdx++)
    {
        int2 loc = int2(posPrev) + offset[sampleIdx];
        if (v[sampleIdx])
        {
            uint sampleVis = gPrevVisTex[loc];
            for (lightIndex = 0; lightIndex < lightCount; lightIndex++)
            {   
                lights[lightIndex] += (1 & sampleVis);
                sampleVis = sampleVis >> 1;
            }
            sumw += w[sampleIdx];
        }       
    }
    
    // redistribute weights in case not all taps were used
    bool valid = (sumw >= 0.01);
    
    if (!valid)
        return;
    
    for (lightIndex = 0; lightIndex < lightCount; lightIndex++)
    {
        int bit = lights[lightIndex] / sumw > 0.5 ? 1 : 0;
        bit = bit << lightIndex;
        prevVisTex += bit;
    }
}

void loadPrevVis3x3(int2 iposPrev, float4 depth, float Zprev,
                    float3 normal, float3 normalPrev, float normalFwidth,
                    Texture2D gPrevVisTex, inout uint prevVisTex)
{
    float cnt = 0.0;

    // this code performs a binary decision for each tap of the cross-bilateral filter
    const int radius = 1;
    const uint lightCount = gScene.getLightCount();
    float lights[32]; // Accumulates interpolated light values
    
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
            int2 p = iposPrev + int2(xx, yy);

            if (isReprjValid(iposPrev, depth.x, Zprev, depth.y, normal, normalPrev, normalFwidth))
            {
                uint sampleVis = gPrevVisTex[p];
                for (lightIndex = 0; lightIndex < lightCount; lightIndex++)
                {
                    lights[lightIndex] += (1 & sampleVis);
                    sampleVis = sampleVis >> 1;
                }
                cnt += 1.0;
            }
        }
    }
        
    if (cnt > 0)
    {
        for (lightIndex = 0; lightIndex < lightCount; lightIndex++)
        {
            int bit = lights[lightIndex] > cnt / 2 ? 1 : 0;
            bit = bit << lightIndex;
            prevVisTex += bit;
        }
    }
}
#endif
