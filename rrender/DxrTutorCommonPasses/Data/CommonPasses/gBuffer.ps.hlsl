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

// Falcor / Slang imports to include shared code and data structures
import Scene.Raster;            // Imports ShaderCommon and DefaultVS, plus material evaluation, VertexOut declaration
import Scene.Scene;

#include "packingUtils.hlsli"

struct GBuffer
{
    float4 wsPos    : SV_Target0;   // World space position.  .w component = 0 if a background pixel
    float4 wsNorm   : SV_Target1;   // World space normal.  (.w is distance from camera to hit point; this may not be used)
    // This render target stores material texture data in a packed format with 8 bits per component.
    // r: diffuse.r,    diffuse.g,      diffuse.b,          opacity
    // g: specular.r,   specular.g,     specular.b,         linear roughness
    // b: emissive.r,   emissive.g,     emissive.b,         doubleSided ? 1.0f : 0.0f
    // a: IoR,          metallic,       specular trans,     eta
    float4 texData  : SV_Target2;
};

// Our main entry point for the g-buffer fragment shader.
GBuffer main(VSOut vsOut, uint primID : SV_PrimitiveID)
{
    float3 cameraPosW = gScene.camera.getPosition();
    float3 viewDir = normalize(cameraPosW - vsOut.posW);
    // This is a Falcor built-in that extracts data suitable for shading routines
    //     (see ShaderCommon.slang for the shading data structure and routines)
    ShadingData hitPt = prepareShadingData(vsOut, primID, viewDir);

    // Check if we hit the back of a double-sided material, in which case, we flip
    //     normals around here (so we don't need to when shading)
    float NdotV = dot(normalize(hitPt.N), viewDir);
    if (NdotV <= 0.0f && hitPt.isDoubleSided())
        hitPt.N = -hitPt.N;

    // Dump out our G buffer channels
    GBuffer gBufOut;
    gBufOut.wsPos    = float4(hitPt.posW, 1.f);
    gBufOut.wsNorm   = float4(hitPt.N, length(hitPt.posW - cameraPosW));
    // Use the function in packingUtils.hlsli to extract the texture data in a compact format
    gBufOut.texData = asfloat(packTextureData(hitPt));
    return gBufOut;
}


