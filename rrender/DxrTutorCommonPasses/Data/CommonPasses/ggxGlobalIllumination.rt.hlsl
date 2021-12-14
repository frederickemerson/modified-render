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

#include "packingUtils.hlsli"              // Functions used to unpack the GBuffer's gTexData
#include "Utils/Math/MathConstants.slangh"

// Include and import common Falcor utilities and data structures
import Scene.Raytracing;
import Scene.Shading;                      // Shading functions, etc   
import Scene.Lights.Lights;                // Light structures for our current scene

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
shared RWTexture2D<float4> gOutput;

// A separate file with some simple utility functions: getPerpendicularVector(), initRand(), nextRand()
#include "ggxGlobalIlluminationUtils.hlsli"

// Include implementations of GGX normal distribution function, Fresnel approx,
//     masking term and function to sampl NDF 
#include "microfacetBRDFUtils.hlsli"

// Include shader entries, data structures, and utility functions to spawn rays
#include "standardShadowRay.hlsli"
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
    //if (dot(noMapN, V) <= 0.0f) noMapN = -noMapN;

    // If we don't hit any geometry, our difuse material contains our background color.
    float3 shadeColor    = isGeometryValid ? float3(0,0,0) : difMatlColor.rgb;

    // Initialize our random number generator
    uint randSeed        = initRand(launchIndex.x + launchIndex.y * launchDim.x, gFrameCount, 16);

    // Do shading, if we have geoemtry here (otherwise, output the background color)
    if (isGeometryValid)
    {
        // Add any emissive color from primary rays
        shadeColor = gEmitMult * pixelEmissive.rgb;

        // (Optionally) do explicit direct lighting to a random light in the scene
        if (gDoDirectGI)
        {
            // Compute the incoming direct illumination from a random light, and albedo of the hit spot
            float3 directColor, directAlbedo;
            ggxDirect(randSeed, worldPos.xyz, worldNorm.xyz, V, difMatlColor.rgb, specMatlColor.rgb, roughness,
                      directColor, directAlbedo);
            // Don't add if there are any nan values
            float3 addition = directColor * directAlbedo;
            shadeColor += any(isnan(addition)) ? float3(0, 0, 0) : addition;
        }
            

        // (Optionally) do indirect lighting for global illumination
        if (gDoIndirectGI && (gMaxDepth > 0))
        {
            // Compute the incoming indirect illumination either from the diffuse or GGX lobe
            float3 indirectColor, indirectAlbedo;
            ggxIndirect(randSeed, worldPos.xyz, worldNorm.xyz, noMapN,
                V, difMatlColor.rgb, specMatlColor.rgb, roughness, 0, indirectColor, indirectAlbedo);
            // Don't add if there are any nan values
            float3 addition = indirectColor * indirectAlbedo;
            shadeColor += any(isnan(addition)) ? float3(0, 0, 0) : addition;
        }
            
    }
    
    // Since we didn't do a good job above catching NaN's, div by 0, infs, etc.,
    //    zero out samples that would blow up our frame buffer.  Note:  You can and should
    //    do better, but the code gets more complex with all error checking conditions.
    bool colorsNan = any(isnan(shadeColor));

    // Store out the color of this shaded pixel
    gOutput[launchIndex] = float4(colorsNan?float3(0,0,0):shadeColor, 1.0f);
}
