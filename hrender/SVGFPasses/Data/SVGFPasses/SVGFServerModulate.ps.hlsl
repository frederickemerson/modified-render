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

#include "SVGFCommon.hlsli"
#include "../../../DxrTutorCommonPasses/Data/CommonPasses/packingUtils.hlsli"  // Utilities to pack the GBuffer content

// Demodulated input textures from the shading pass
Texture2D<float4> gIndirect;
Texture2D<float4> gIndirAlbedo;

// Texture data from the shading pass - we need this to retrieve the emissive color
Texture2D<float4> gTexData;

// Output texture
RWTexture2D<uint4> gOutput;

bool gFilterEnabled;

// We could directly output a float4 instead of this PS_OUT, but this allows for
// future extension if necessary.
struct PS_OUT
{
    float4 color : SV_TARGET0;
};

float4 main(FullScreenPassVsOut vsOut)
{
    float4 fragCoord = vsOut.posH;
    int2 ipos        = int2(fragCoord.xy);

    PS_OUT ret;
    ret.color = gIndirect[ipos] * gIndirAlbedo[ipos];
    
    // Albedo texture contains light index in alpha.
    uint4 shadeIndirect = ret.color * 255.0;
    gOutput[ipos] = uint4(shadeIndirect.rgb, gIndirAlbedo[ipos].a);
    
    // If filter not enabled, output buffer is not used. So we just return float4(0.0f)
    return gFilterEnabled ? ret.color : float4(0.0f);
}

