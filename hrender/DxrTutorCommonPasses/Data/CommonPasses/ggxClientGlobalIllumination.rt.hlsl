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

// A constant buffer we'll populate from our C++ code  (used for our ray generation shader)
shared cbuffer GlobalCB
{
    uint  gFrameCount;     // An integer changing every frame to update the random number
    bool  gDoIndirectGI;   // A boolean determining if we should shoot indirect GI rays
    bool  gDoDirectGI;     // A boolean determining if we should compute direct lighting
    float gEmitMult;       // Multiply emissive amount by this factor (set to 1, usually)
}

// Input and out textures that need to be set by the C++ code (for the ray gen shader)
shared Texture2D<float4>   gPos;
shared Texture2D<float4>   gNorm;
shared Texture2D<float4>   gTexData;
shared Texture2D<float4>   gEnvMap;
shared Texture2D<uint4>    gGIData;
shared RWTexture2D<float4> gDirectColorOutput;
shared RWTexture2D<float4> gDirectAlbedoOutput;
shared RWTexture2D<float4> gIndirectLightOut;

// A separate file with some simple utility functions: getPerpendicularVector(), initRand(), nextRand()
#include "ggxGlobalIlluminationUtils.hlsli"

// Include implementations of GGX normal distribution function, Fresnel approx,
//     masking term and function to sampl NDF 
#include "microfacetBRDFUtils.hlsli"

//ggxDirect function which uses the given light index instead of sampling for one.
void ggxDirectWithIdx(int lightIdx, float3 hit, float3 N, float3 V, float3 dif, float3 spec, float rough,
    out float3 directColor, out float3 directAlbedo)
{
    // Occluded from the selected light. No contribution for direct illumination.
    if (lightIdx == 0xFF)
    {
        directColor = float3(0.0f);
        directAlbedo = float3(0.0f);
        return;
    }
    
    // Get the number of lights in the scene
    const uint lightCount = gScene.getLightCount();
      
    // Query the scene to find info about the randomly selected light
    float distToLight;
    float3 lightIntensity;
    float3 L;
    getLightData(lightIdx, hit, L, lightIntensity, distToLight);

    // Compute our lambertion term (N dot L)
    float NdotL = saturate(dot(N, L));

    // Shoot our shadow ray to our randomly selected light.
    float shadowMult = float(lightCount);

    // Compute our GGX color (NdotL term is already cancelled to avoid division by 0).
    float3 ggxTerm = getGGXColor(V, L, N, spec, rough);

    // Compute our final color (combining diffuse lobe plus specular GGX lobe)
    directColor = shadowMult * lightIntensity;
    directAlbedo = ggxTerm + NdotL * dif / M_PI;

    bool colorsNan = any(isnan(directColor)) || any(isnan(directAlbedo));
    directColor = colorsNan ? float3(0.f, 0.f, 0.f) : directColor;
    directAlbedo = colorsNan ? float3(0.f, 0.f, 0.f) : directAlbedo;
}

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
    
    // Does this g-buffer pixel contain a valid piece of geometry?  (0 in pos.w for invalid)
    bool isGeometryValid = (worldPos.w != 0.0f);

    // Extract and compute some material and geometric parameters
    float roughness = specMatlColor.a * specMatlColor.a;
    float3 V        = normalize(gScene.camera.getPosition() - worldPos.xyz);

    // Compute the incoming direct illumination from a random light, and albedo of the hit spot
    float3 outDirectColor, outDirectAlbedo;
    
    // Direct color and albedo have to be computed. Indirect lighting has been computed on the server.
    float3 indirectColor = float3(gGIData[launchIndex].rgb) / 255.0f;
    
    // Initialize our random number generator
    uint randSeed = initRand(launchIndex.x + launchIndex.y * launchDim.x, gFrameCount, 16);

    // Do shading, if we have geoemtry here (otherwise, output the background color)
    if (isGeometryValid)
    {
        // Do explicit direct lighting to a random light in the scene
        int lightToSample = gGIData[launchIndex].a;
          
        ggxDirectWithIdx(lightToSample, worldPos.xyz, worldNorm.xyz, V, difMatlColor.rgb, specMatlColor.rgb, roughness,
                    outDirectColor, outDirectAlbedo);
            
        outDirectColor = any(isnan(indirectColor)) ? float3(0.0f) : outDirectColor;
    }

    // We output direct color and albedo separately for the SVGF pass.
    gDirectColorOutput[launchIndex] = float4(outDirectColor, 1.0);
    gDirectAlbedoOutput[launchIndex] = float4(outDirectAlbedo, 1.0);
    gIndirectLightOut[launchIndex] = float4(indirectColor, 1.0f);
}


