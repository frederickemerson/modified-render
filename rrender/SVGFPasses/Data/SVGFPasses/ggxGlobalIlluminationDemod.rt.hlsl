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

#include "../../../DxrTutorCommonPasses/Data/CommonPasses/packingUtils.hlsli"  // Utilities to pack the GBuffer content
#include "Utils/Math/MathConstants.slangh"

// Include and import common Falcor utilities and data structures
import Scene.Raytracing;                   // Shared ray tracing specific functions & data 
import Scene.Shading;                      // Shading functions, etc   
import Scene.Lights.Lights;                // Light structures for our current scene

// A separate file with some simple utility functions: getPerpendicularVector(), initRand(), nextRand()
#include "ggxGlobalIlluminationDemodUtils.hlsli"
#include "../../../DxrTutorCommonPasses/Data/CommonPasses/ggxGlobalIlluminationUtils.hlsli"

// Include implementations of GGX normal distribution function, Fresnel approx,
//     masking term and function to sampl NDF 
#include "../../../DxrTutorCommonPasses/Data/CommonPasses//microfacetBRDFUtils.hlsli"

// Include shader entries, data structures, and utility functions to spawn rays
#include "../../../DxrTutorCommonPasses/Data/CommonPasses//standardShadowRay.hlsli"
#include "../../../DxrTutorCommonPasses/Data/CommonPasses//indirectRay.hlsli"

// A constant buffer we'll populate from our C++ code  (used for our ray generation shader)
cbuffer RayGenCB
{
    float gMinT;           // Min distance to start a ray to avoid self-occlusion
    uint  gFrameCount;     // An integer changing every frame to update the random number
    bool  gDoIndirectGI;   // A boolean determining if we should shoot indirect GI rays
    bool  gDoDirectGI;     // A boolean determining if we should compute direct lighting
    uint  gMaxDepth;       // Maximum number of recursive bounces to allow
    float gEmitMult;       // Multiply emissive amount by this factor (set to 1, usually)
}

// Input textures that need to be set by the C++ code (for the ray gen shader)
Texture2D<float4> gPos;
Texture2D<float4> gNorm;
Texture2D<float4> gTexData;

// Output textures that need to be set by the C++ code (for the ray gen shader)
RWTexture2D<float4> gDirectOut;
RWTexture2D<float4> gIndirectOut;
RWTexture2D<float4> gOutAlbedo;
RWTexture2D<float4> gIndirAlbedo;

// Input and out textures that need to be set by the C++ code (for the miss shader)
Texture2D<float4> gEnvMap;

[shader("raygeneration")]
void SimpleDiffuseGIRayGen()
{
    // Where is this ray on screen?
    uint2 launchIndex = DispatchRaysIndex().xy;
    uint2 launchDim   = DispatchRaysDimensions().xy;

    // Load g-buffer data
    float4 worldPos      = gPos[launchIndex];
    float4 worldNorm     = gNorm[launchIndex];
    // Get the texture data that is stored in a compact format
    float4 difMatlColor;
    float4 specMatlColor;
    float4 pixelEmissive;
    float4 matlOthers;
    unpackTextureData(asuint(gTexData[launchIndex]), difMatlColor, specMatlColor, pixelEmissive, matlOthers);
    // pixelEmissive.w is 1.f if material is double sided, 0.f otherwise
    float4 extraData = float4(0.0f, pixelEmissive.w, 0.0f, 0.0f);

    // Does this g-buffer pixel contain a valid piece of geometry?  (0 in pos.w for invalid)
    bool isGeometryValid = (worldPos.w != 0.0f);

    // Extract some material parameters
    float roughness      = specMatlColor.a * specMatlColor.a;
    float3 toCamera      = normalize(gScene.camera.getPosition() - worldPos.xyz);

    // Grab our geometric normal.  Also make sure this points the right direction.
    //     This is badly hacked into our G-buffer for now.  We need this because 
    //     sometimes, when normal mapping, our randomly selected indirect ray will 
    //     be *below* the surface (due to the normal map perturbations), which will 
    //     cause light leaking.  We solve by ignoring the ray's contribution if it
    //     is below the horizon.  
    float3 noMapN = normalize(extraData.yzw);
    //if (dot(noMapN, toCamera) <= 0.0f) noMapN = -noMapN;

    // Initialize our random number generator
    uint randSeed = initRand(launchIndex.x + launchIndex.y * launchDim.x, gFrameCount, 16);

    // Do shading, if we have geoemtry here (otherwise, output the background color)
    if (isGeometryValid)
    {
        // (Optionally) do explicit direct lighting to a random light in the scene
        if (gDoDirectGI)
        {
            // Compute the incoming direct illumination from a random light, and albedo of the hit spot
            float3 directColor, directAlbedo;
            ggxDirect(randSeed, worldPos.xyz, worldNorm.xyz, toCamera, difMatlColor.rgb, specMatlColor.rgb, roughness,
                      directColor, directAlbedo);
            // Store the results
            gDirectOut[launchIndex] = float4(directColor, 1.0f);
            gOutAlbedo[launchIndex] = float4(directAlbedo, 1.0f);
        }

        // (Optionally) do indirect lighting for global illumination
        if (gDoIndirectGI && (gMaxDepth > 0))
        {
            // Compute the incoming indirect illumination either from the diffuse or GGX lobe
            float3 indirectColor, indirectAlbedo;
            ggxIndirect(randSeed, worldPos.xyz, worldNorm.xyz, noMapN,
                toCamera, difMatlColor.rgb, specMatlColor.rgb, roughness, 0, indirectColor, indirectAlbedo);
            // Store the results
            gIndirectOut[launchIndex] = float4(indirectColor, 1.0f);
            gIndirAlbedo[launchIndex] = float4(indirectAlbedo, 1.0f);
        }
    }
    else
    {
        // If we hit the background color, return reasonable values that won't mess up the SVGF filter
        gDirectOut[launchIndex] = float4(difMatlColor.rgb, 1.0f);    // DifMatlColor is the env. map color, in this case
        gIndirectOut[launchIndex] = float4(0.0f, 0.0f, 0.0f, 1.0f);
        gOutAlbedo[launchIndex] = float4(1.0f, 1.0f, 1.0f, 1.0f);
    }
}
