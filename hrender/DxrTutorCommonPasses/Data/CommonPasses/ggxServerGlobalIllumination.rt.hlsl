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
#pragma once

#include "packingUtils.hlsli"              // Functions used to unpack the GBuffer's gTexData
#include "Utils/Math/MathConstants.slangh"

// Include and import common Falcor utilities and data structures
import Scene.Raytracing;
import Scene.Shading;                      // Shading functions, etc   

// A constant buffer we'll populate from our C++ code  (used for our ray generation shader)
shared cbuffer GlobalCB
{
    float gMinT;           // Min distance to start a ray to avoid self-occlusion
    uint  gFrameCount;     // An integer changing every frame to update the random number
    bool  gDoIndirectGI;   // A boolean determining if we should shoot indirect GI rays
    bool  gDoDirectGI;     // A boolean determining if we should compute direct lighting
    uint  gMaxDepth;       // Maximum number of recursive bounces to allow
    float gEmitMult;       // Multiply emissive amount by this factor (set to 1, usually)
}

// Input and out textures that need to be set by the C++ code (for the ray gen shader)
shared Texture2D<float4>   gPos;
shared Texture2D<float4>   gNorm;
shared Texture2D<float4>   gTexData;
shared Texture2D<float4>   gEnvMap;
shared RWTexture2D<float4> gColorOutput;
shared RWTexture2D<float4> gAlbedoOutput;

// A separate file with some simple utility functions: getPerpendicularVector(), initRand(), nextRand()
#include "ggxGlobalIlluminationUtils.hlsli"

// Include implementations of GGX normal distribution function, Fresnel approx,
//     masking term and function to sampl NDF 
#include "microfacetBRDFUtils.hlsli"

// Include shader entries, data structures, and utility functions to spawn rays
#include "indirectRay.hlsli"

// How do we shade our g-buffer and spawn indirect and shadow rays?
[shader("raygeneration")]
void SimpleDiffuseGIRayGen()
{
    // Where is this ray on screen?
    uint2 launchIndex    = DispatchRaysIndex().xy;
    uint2 launchDim      = DispatchRaysDimensions().xy;

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

    // Extract and compute some material and geometric parameters
    float roughness = specMatlColor.a * specMatlColor.a;
    float3 V        = normalize(gScene.camera.getPosition() - worldPos.xyz);

    // Grab our geometric normal.  Also make sure this points the right direction.
    //     This is badly hacked into our G-buffer for now.  We need this because 
    //     sometimes, when normal mapping, our randomly selected indirect ray will 
    //     be *below* the surface (due to the normal map perturbations), which will 
    //     cause light leaking.  We solve by ignoring the ray's contribution if it
    //     is below the horizon.  
    float3 noMapN = normalize(extraData.yzw);
    if (dot(noMapN, V) <= 0.0f)
        noMapN = -noMapN;

    // If we don't hit any geometry, our diffuse material contains our background color.
    float3 shadeColor    = isGeometryValid ? float3(0.0f) : difMatlColor.rgb;

    // By default, 255 signifies no direct light sampling.
    uint lightToSample = 0xFF;
    
    // Initialize our random number generator
    uint randSeed        = initRand(launchIndex.x + launchIndex.y * launchDim.x, gFrameCount, 16);

    // Store our indirect color computation
    float3 indirectColor = shadeColor;
    float3 indirectAlbedo = float3(1.0);
    
    // Do shading, if we have geometry here (otherwise, output the background color)
    if (isGeometryValid)
    {
        // (Optionally) do explicit direct lighting to a random light in the scene
        if (gDoDirectGI)
        {     
            /* On the server side, we simply select the light we want to use.
               The lighting computation will be done in the client. */
            
            // Get the number of lights in the scene
            const uint lightCount = gScene.getLightCount();
            
            // Pick a random light index from our scene to shoot a shadow ray towards
            lightToSample = min(int(nextRand(randSeed) * lightCount), lightCount - 1);
            
            float distToLight;
            float3 lightIntensity;
            float3 L;
            float3 hit = worldPos.xyz;
            getLightData(lightToSample, hit, L, lightIntensity, distToLight);
            
            // If hit point is occluded, we mark the light index as 0xFF, signifying occlusion
            lightToSample = shadowRayVisibility(hit, L, gMinT, distToLight) == 1.0f ? lightToSample : 0xFF;
        }
        else // No shadow calculation, so we don't occlude any light.
        {           
            // Get the number of lights in the scene
            const uint lightCount = gScene.getLightCount();
            
            // Pick a random light index from our scene to shoot a shadow ray towards
            lightToSample = min(int(nextRand(randSeed) * lightCount), lightCount - 1);
        }
            
        // (Optionally) do indirect lighting for global illumination
        if (gDoIndirectGI && (gMaxDepth > 0))
        {
            // We have to raytrace for indirect illumination, so we send the computed indirect colour to the client.
            // Compute the incoming indirect illumination either from the diffuse or GGX lobe

            ggxIndirect(randSeed, worldPos.xyz, worldNorm.xyz, noMapN,
                V, difMatlColor.rgb, specMatlColor.rgb, roughness, 0, indirectColor, indirectAlbedo);
            
            // Don't add if there are any nan values
            indirectColor = any(isnan(indirectColor)) ? float3(0.0f) : indirectColor;
        }
            
    }
    else
    {
        // If we hit the background color, return reasonable values that won't mess up the SVGF filter
        indirectColor = float3(0.0f);
        indirectAlbedo = float3(1.0f);
    }

    // Store out the color of this shaded pixel
    gColorOutput[launchIndex] = float4(indirectColor, 1.0);
    gAlbedoOutput[launchIndex] = float4(indirectAlbedo, lightToSample);
}
