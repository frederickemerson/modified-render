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
#include "../../../DxrTutorCommonPasses/Data/CommonPasses/packingUtils.hlsli"  // Utilities to pack the GBuffer content

Texture2D<uint>   gVis;
Texture2D<uint>   gAo;
Texture2D<float2>   gVar;
Texture2D<float4>   gCompactNormDepth;
Texture2D<float4>   gHistoryLength;

cbuffer PerImageCB : register(b0)
{
    int         gStepSize;
    float       gPhiColor;
    float       gPhiNormal;
};

// computes a 3x3 gaussian blur of the variance, centered around
// the current pixel
float2 computeVarianceCenter(int2 ipos, Texture2D<uint> sAo)
{
    float2 sum = float2(0.0, 0.0);

    const float kernel[2][2] = {
        { 1.0 / 4.0, 1.0 / 8.0  },
        { 1.0 / 8.0, 1.0 / 16.0 }
    };

    //const uint lightCount = 32;
    //float lights[32]; // Accumulates interpolated light values
    
    //int lightIndex;
    
    //// Just in case not initialized, can remove if confirmed
    //for (lightIndex = 0; lightIndex < lightCount; lightIndex++)
    //{
    //    lights[lightIndex] = 0;
    //}
    
    const int radius = 1;
    for (int yy = -radius; yy <= radius; yy++)
    {
        for (int xx = -radius; xx <= radius; xx++)
        {
            int2 p = ipos + int2(xx, yy);

            float k = kernel[abs(xx)][abs(yy)];

            const uint ao = sAo.Load(int3(p, 0));
            //uint visVar = asuint(var.r);
            
            //for (lightIndex = 0; lightIndex < lightCount; lightIndex++)
            //{
            //    lights[lightIndex] += (1 & visVar) * k;
            //    visVar = visVar >> 1;
            //}
            
            sum.g += ao * k;
        }
    }

    //uint sumVis = 0;
    //for (lightIndex = 0; lightIndex < lightCount; lightIndex++)
    //{
    //    int bit = lights[lightIndex] > 0.5 ? 1 : 0;
    //    bit = bit << lightIndex;
    //    sumVis += bit;
    //}
    
    //sum.r = asfloat(sumVis);
    return sum;
}

struct PS_OUT
{
    uint OutVis    : SV_TARGET0;
    uint OutAo  : SV_TARGET1;
    float2 OutVar : SV_TARGET2;
};

PS_OUT main(FullScreenPassVsOut vsOut)
{

    float4 fragCoord = vsOut.posH;
    const int2 ipos       = int2(fragCoord.xy);
    const int2 screenSize = getTextureDims(gHistoryLength, 0);

    const float epsVariance      = 1e-10;
    const float kernelWeights[3] = { 1.0, 2.0 / 3.0, 1.0 / 6.0 };

    // constant samplers to prevent the compiler from generating code which
    // fetches the sampler descriptor from memory for each texture access
    const uint  visCenter    = gVis.Load(int3(ipos, 0));
    const uint  aoCenter  = gAo.Load(int3(ipos, 0));

    // variance for vis and ao, filtered using 3x3 gaussin blur
    const float2 var = computeVarianceCenter(ipos, gAo);
    
    
    // number of temporally integrated pixels
    const float historyLength = gHistoryLength.Load(int3(ipos, 0)).r;

    float3 normalCenter;
    float2 zCenter;
    fetchNormalAndLinearZ(gCompactNormDepth, ipos, normalCenter, zCenter);

    PS_OUT psOut;
    
    if (zCenter.x < 0)
    {
        // not a valid depth => must be envmap => do not filter
        psOut.OutVis = visCenter;
        psOut.OutAo = aoCenter;
        psOut.OutVar = gVar.Load(int3(ipos, 0));
        return psOut;
    }

    const float phiAo = gPhiColor * sqrt(max(0.0, epsVariance + var.g));
    const float phiDepth = max(zCenter.y, 1e-8) * gStepSize;

    // explicitly store/accumulate center pixel with weight 1 to prevent issues
    // with the edge-stopping functions
    float sumW = 1.0;
    uint sumVis = visCenter;
    float sumAo = aoCenter;

    for (int yy = -2; yy <= 2; yy++)
    {
        for (int xx = -2; xx <= 2; xx++)
        {
            const int2 p = ipos + int2(xx, yy) * gStepSize;
            const bool inside = all(p >= int2(0, 0)) && all(p < screenSize);

            const float kernel = kernelWeights[abs(xx)] * kernelWeights[abs(yy)];

            if (inside && (xx != 0 || yy != 0)) // skip center pixel, it is already accumulated
            {
                const uint aoP = gAo.Load(int3(p, 0));

                float3 normalP;
                float2 zP;
                fetchNormalAndLinearZ(gCompactNormDepth, p, normalP, zP);

                const float w = computeWeight(
                        zCenter.x, zP.x, phiDepth * length(float2(xx, yy)),
                        normalCenter, normalP, gPhiNormal,
                        asfloat(aoCenter), asfloat(aoP), phiAo) * kernel;

                // alpha channel contains the variance, therefore the weights need to be squared, see paper for the formula
                sumW += w;
                sumAo += w * aoP;
            }
        }
    }

    // renormalization is different for variance, check paper for the formula
    psOut.OutVis = visCenter | asuint(var.r);
    psOut.OutAo = sumAo / sumW;
    psOut.OutVar = float2(var.x, var.y / (sumW * sumW));
    
    // We copy from the out buffer into the output textures after this shader is run.
    
    return psOut;
}
