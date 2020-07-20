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
import Scene.Raster;        // Imports ShaderCommon and DefaultVS, plus material evaluation
import Scene.Scene;         // VertexOut declaration
import Scene.Camera.Camera;

#include "../../../DxrTutorCommonPasses/Data/CommonPasses/packingUtils.hlsli"  // Utilities to pack the GBuffer content
#include "svgfGBufData.hlsli"  // Our input structure from the vertex shader

// Constant buffer passed down from our C++ code in SVGFPass.cpp
cbuffer GBufCB
{
    float4 gBufSize;  // xy = (size of output buf), zw = 1.0/(size of output buf)
};

// What's in our output G-buffer structure?  This is extremely fat and probably could be cut down, except
//    our research / prototype SVGF filter and simple path tracer uses a bunch of these outputs as full
//    floats.  There's serious room here for G-buffer compression.
struct GBuffer
{
    float4 wsPos       : SV_Target0;   // World space position.  .w component = 0 if a background pixel
    float4 wsNorm      : SV_Target1;   // World space normal.  (.w is distance from camera to hit point; this may not be used)
    // This render target stores material texture data in a packed format with 8 bits per component.
    // r: diffuse.r,    diffuse.g,      diffuse.b,          opacity
    // g: specular.r,   specular.g,     specular.b,         linear roughness
    // b: emissive.r,   emissive.g,     emissive.b,         doubleSided ? 1.0f : 0.0f
    // a: IoR,          metallic,       specular trans,     eta
    float4 texData     : SV_Target2;
    float4 svgfLinZ    : SV_Target3;   // SVGF-specific buffer containing linear z, max z-derivs, last frame's z, obj-space normal
    float4 svgfMoVec   : SV_Target4;   // SVGF-specific buffer containing motion vector and fwidth of pos & normal
    float4 svgfCompact : SV_Target5;   // SVGF-specific buffer containing duplicate data that allows reducing memory traffic in some passes
};
// A simple utility to convert a float to a 2-component octohedral representation packed into one uint
uint dirToOct(float3 normal)
{
    float2 p = normal.xy * (1.0 / dot(abs(normal), 1.0.xxx));
    float2 e = normal.z > 0.0 ? p : (1.0 - abs(p.yx)) * (step(0.0,p)*2.0-(float2)(1.0)); 
    return (asuint(f32tof16(e.y)) << 16) + (asuint(f32tof16(e.x)));
}

// Take current clip position, last frame pixel position and compute a motion vector
float2 calcMotionVector(float4 prevClipPos, float2 currentPixelPos, float2 invFrameSize)
{
    float2 prevPosNDC = (prevClipPos.xy / prevClipPos.w) * float2(0.5, -0.5) + float2(0.5, 0.5);
    float2 motionVec  = prevPosNDC - (currentPixelPos * invFrameSize);

    // Guard against inf/nan due to projection by w <= 0.
    const float epsilon = 1e-5f;
    motionVec = (prevClipPos.w < epsilon) ? float2(0, 0) : motionVec;
    return motionVec;
}

// Our main entry point for the g-buffer fragment shader.
GBuffer main(GBufVertexOut vsOut, uint primID : SV_PrimitiveID)
{
    Camera camera = gScene.camera;

    // Grab shading data.  Invert if necessary.
    float3 cameraPosW = camera.getPosition();
    float3 viewDir = normalize(cameraPosW - vsOut.base.posW);
    ShadingData hitPt = prepareShadingData(vsOut.base, primID, viewDir);

    // Check if we hit the back of a double-sided material, in which case, we flip
    //     normals around here (so we don't need to when shading)
    float NdotV = dot(normalize(hitPt.N), viewDir);
    if (NdotV <= 0.0f && hitPt.doubleSided)
        hitPt.N = -hitPt.N;

    // Compute data needed for SVGF

    // The 'linearZ' buffer
    float linearZ    = vsOut.base.posH.z * vsOut.base.posH.w;
    float maxChangeZ = max(abs(ddx(linearZ)), abs(ddy(linearZ)));
    float objNorm    = asfloat(dirToOct(normalize(vsOut.normalObj)));
    float4 svgfLinearZOut = float4(linearZ, maxChangeZ, vsOut.base.prevPosH.z, objNorm);

    // The 'motion vector' buffer
    float2 svgfMotionVec = calcMotionVector(vsOut.base.prevPosH, vsOut.base.posH.xy, gBufSize.zw) +
                           float2(camera.data.jitterX, -camera.data.jitterY);
    float2 posNormFWidth = float2(length(fwidth(hitPt.posW)), length(fwidth(hitPt.N))); 
    float4 svgfMotionVecOut = float4(svgfMotionVec, posNormFWidth);

    // Dump out our G buffer channels
    GBuffer gBufOut;
    gBufOut.wsPos     = float4(hitPt.posW, 1.f);
    gBufOut.wsNorm    = float4(hitPt.N, length(hitPt.posW - cameraPosW) );
    // Use the function in packingUtils.hlsli to extract the texture data in a compact format
    gBufOut.texData   = asfloat(packTextureData(hitPt));
    gBufOut.svgfLinZ  = svgfLinearZOut;
    gBufOut.svgfMoVec = svgfMotionVecOut;

    // A compacted buffer containing discretizied normal, depth, depth derivative
    gBufOut.svgfCompact = float4( asfloat(dirToOct(hitPt.N)), linearZ, maxChangeZ, 0.0f );

    return gBufOut;
}


