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

#include "microfacetBRDFUtils.hlsli"

// The payload structure for our indirect rays
struct IndirectRayPayload
{
    float3 color;    // The (returned) color in the ray's direction
    uint   rndSeed;  // Our random seed, so we pick uncorrelated RNGs along our ray
    uint   rayDepth; // What is the depth of our current ray?
};

float3 shootIndirectRay(float3 rayOrigin, float3 rayDir, float minT, uint curPathLen, uint seed, uint curDepth)
{
    // Setup our indirect ray
    RayDesc rayColor;
    rayColor.Origin = rayOrigin;  // Where does it start?
    rayColor.Direction = rayDir;  // What direction do we shoot it?
    rayColor.TMin = minT;         // The closest distance we'll count as a hit
    rayColor.TMax = 1.0e38f;      // The farthest distance we'll count as a hit

    // Initialize the ray's payload data with black return color and the current rng seed
    IndirectRayPayload payload;
    payload.color = float3(0, 0, 0);
    payload.rndSeed = seed;
    payload.rayDepth = curDepth + 1;

    // Trace our ray to get a color in the indirect direction.  Use hit group #1 and miss shader #1
    TraceRay(gRtScene, 0, 0xFF, 1, hitProgramCount, 1, rayColor, payload);

    // Return the color we got from our ray
    return payload.color;
}

[shader("miss")]
void IndirectMiss(inout IndirectRayPayload rayData)
{
    // Load some information about our lightprobe texture
    float2 dims;
    gEnvMap.GetDimensions(dims.x, dims.y);

    // Convert our ray direction to a (u,v) coordinate
    float2 uv = wsVectorToLatLong(WorldRayDirection());

    // Load our background color, then store it into our ray payload
    rayData.color = gEnvMap[uint2(uv * dims)].rgb;
}

[shader("anyhit")]
void IndirectAnyHit(uniform HitShaderParams hitParams, inout IndirectRayPayload rayData, BuiltInTriangleIntersectionAttributes attribs)
{
    // Is this a transparent part of the surface?  If so, ignore this hit
    if (alphaTestFails(hitParams, attribs))
        IgnoreHit();
}

float3 lambertianDirect(inout uint rndSeed, float3 hit, float3 norm, float3 difColor)
{
    // Get the number of lights in the scene
    const uint lightCount = gScene.getLightCount();

    // Pick a random light from our scene to shoot a shadow ray towards
    int lightToSample = min(int(nextRand(rndSeed) * lightCount), lightCount - 1);

    // Query the scene to find info about the randomly selected light
    float distToLight;
    float3 lightIntensity;
    float3 toLight;
    getLightData(lightToSample, hit, toLight, lightIntensity, distToLight);

    // Compute our lambertion term (L dot N)
    float LdotN = saturate(dot(norm, toLight));

    // Shoot our shadow ray to our randomly selected light
    float shadowMult = float(lightCount) * shadowRayVisibility(hit, toLight, gMinT, distToLight);

    // Return the Lambertian shading color using the physically based Lambertian term (albedo / pi)
    return shadowMult * LdotN * lightIntensity * difColor / M_PI;
}

float3 lambertianIndirect(inout uint rndSeed, float3 hit, float3 norm, float3 difColor, uint rayDepth)
{
    // Shoot a randomly selected cosine-sampled diffuse ray.
    float3 L = getCosHemisphereSample(rndSeed, norm);
    float3 bounceColor = shootIndirectRay(hit, L, gMinT, 0, rndSeed, rayDepth);

    // Accumulate the color: (NdotL * incomingLight * difColor / pi) 
    // Probability of sampling:  (NdotL / pi)
    return bounceColor * difColor;
}

// Get the direct illumination and albedo of a hit spot by randomly sampling a light and computing the GGX BRDF.
void ggxDirect(inout uint rndSeed, float3 hit, float3 N, float3 V, float3 dif, float3 spec, float rough,
    out float3 directColor, out float3 directAlbedo)
{
    // Get the number of lights in the scene
    const uint lightCount = gScene.getLightCount();

    // Pick a random light from our scene to shoot a shadow ray towards
    int lightToSample = min(int(nextRand(rndSeed) * lightCount), lightCount - 1);

    // Query the scene to find info about the randomly selected light
    float distToLight;
    float3 lightIntensity;
    float3 L;
    getLightData(lightToSample, hit, L, lightIntensity, distToLight);

    // Compute our lambertion term (N dot L)
    float NdotL = saturate(dot(N, L));

    // Shoot our shadow ray to our randomly selected light
    float shadowMult = float(lightCount) * shadowRayVisibility(hit, L, gMinT, distToLight);

    // Compute our GGX color
    float3 ggxTerm = getGGXColor(V, L, N, spec, rough);

    // Compute our final color (combining diffuse lobe plus specular GGX lobe)
    directColor = shadowMult * lightIntensity * NdotL;
    directAlbedo = ggxTerm + dif / M_PI;

    bool colorsNan = any(isnan(directColor)) || any(isnan(directAlbedo));
    directColor = colorsNan ? float3(0.f, 0.f, 0.f) : directColor;
    directAlbedo = colorsNan ? float3(0.f, 0.f, 0.f) : directAlbedo;
}

void ggxIndirect(inout uint rndSeed, float3 hit, float3 N, float3 noNormalN, float3 V, float3 dif, float3 spec, float rough, uint rayDepth,
    out float3 indirectColor, out float3 indirectAlbedo)
{
    // We have to decide whether we sample our diffuse or specular/ggx lobe.
    float probDiffuse = probabilityToSampleDiffuse(dif, spec);
    float chooseDiffuse = (nextRand(rndSeed) < probDiffuse);

    indirectAlbedo = /* NdotL * */ max (5e-3f, dif / M_PI);

    // If we randomly selected to sample our diffuse lobe...
    if (chooseDiffuse)
    {
        // Shoot a randomly selected cosine-sampled diffuse ray.
        float3 L = getCosHemisphereSample(rndSeed, N);
        float3 bounceColor = shootIndirectRay(hit, L, gMinT, 0, rndSeed, rayDepth);

        // Check to make sure our randomly selected, normal mapped diffuse ray didn't go below the surface.
        //if (dot(noNormalN, L) <= 0.0f) bounceColor = float3(0, 0, 0);

        // Accumulate the color: (NdotL * incomingLight * dif / pi) 
        // Probability of sampling:  (NdotL / pi) * probDiffuse
        indirectColor = /* NdotL * */ bounceColor * M_PI / probDiffuse;
    }
    // Otherwise we randomly selected to sample our GGX lobe
    else
    {
        float3 ggxBRDF, L; float ggxProb, NdotL;
        getGGXColorAndProb(rndSeed, V, N, spec, rough, ggxBRDF, ggxProb, L, NdotL);

        // Compute our color by tracing a ray in this direction
        float3 bounceColor = shootIndirectRay(hit, L, gMinT, 0, rndSeed, rayDepth);

        // Check to make sure our randomly selected, normal mapped diffuse ray didn't go below the surface.
        //if (dot(noNormalN, L) <= 0.0f) bounceColor = float3(0, 0, 0);

        // Accumulate the color:  ggx-BRDF * incomingLight * NdotL / probability-of-sampling
        //    -> Should really simplify the math above.
        indirectColor = NdotL * bounceColor * ggxBRDF / (indirectAlbedo * ggxProb * (1.0f - probDiffuse));
        //return NdotL * bounceColor * ggxBRDF / (ggxProb * (1.0f - probDiffuse));
    }

    bool colorsNan = any(isnan(indirectColor)) || any(isnan(indirectAlbedo));
    indirectColor = colorsNan ? float3(0.f, 0.f, 0.f) : indirectColor;
}

[shader("closesthit")]
void IndirectClosestHit(uniform HitShaderParams hitParams, inout IndirectRayPayload rayData, BuiltInTriangleIntersectionAttributes attribs)
{
    // Run a helper functions to extract Falcor scene data for shading
    ShadingData shadeData = getHitShadingData( hitParams, attribs, WorldRayOrigin() );

    // Add emissive color
    rayData.color = gEmitMult * shadeData.emissive.rgb;

    // Do direct illumination at this hit location
    if (gDoDirectGI)
    {
        float3 directColor, directAlbedo;
        ggxDirect(rayData.rndSeed, shadeData.posW, shadeData.N, shadeData.V,
            shadeData.diffuse, shadeData.specular, shadeData.linearRoughness, directColor, directAlbedo);
        rayData.color += directColor * directAlbedo;
    }

    // Do indirect illumination at this hit location (if we haven't traversed too far)
    if (rayData.rayDepth < gMaxDepth)
    {
        // Use the same normal for the normal-mapped and non-normal mapped vectors... This means we could get light
        //     leaks at secondary surfaces with normal maps due to indirect rays going below the surface.  This
        //     isn't a huge issue, but this is a (TODO: fix)
        float3 indirectColor, indirectAlbedo;
        ggxIndirect(rayData.rndSeed, shadeData.posW, shadeData.N, shadeData.N, shadeData.V,
            shadeData.diffuse, shadeData.specular, shadeData.linearRoughness, rayData.rayDepth, indirectColor, indirectAlbedo);
        rayData.color += indirectColor * indirectAlbedo;
    }
}
