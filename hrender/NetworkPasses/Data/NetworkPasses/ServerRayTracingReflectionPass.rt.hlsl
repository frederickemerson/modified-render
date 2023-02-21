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

// Include and import common Falcor utilities and data structures
import Scene.Raytracing;
import Scene.Shading;                          // Shading functions, etc
import Experimental.Scene.Lights.LightHelpers; // Light structures for our current scene
import Utils.Color.ColorHelpers;

#include "../../../DxrTutorCommonPasses/Data/CommonPasses/packingUtils.hlsli" // Functions used to unpack the GBuffer's gTexData
#include "networkUtils.hlsli" // Some simple utility functions: getPerpendicularVector(), initRand(), nextRand()
#include "../../../DxrTutorCommonPasses/Data/CommonPasses/microfacetBRDFUtils.hlsli"

// Payload for our GGX ray.
struct GGXPayload
{
    float3 color; // The (returned) color in the ray's direction
};

// A constant buffer we'll fill in for our ray generation shader
cbuffer RayGenCB
{
    float gMinT;
    bool gSkipSRT; // Skip this GGX pass
    float gLumThreshold; // Threshold for luminance of reflected light
    float gRoughnessThreshold; // Threshold for roughness
    bool gUseThresholds;
}

// Input and out textures that need to be set by the C++ code
Texture2D<float4>   gPos;
Texture2D<float4>   gNorm;
Texture2D<float4>   gTexData;
Texture2D<float4>   gVshading;
Texture2D<uint>     gVisibility;
Texture2D<uint>     gRaymask;
RWTexture2D<float3> gOutput;

float3 shootGGXray(float3 origin, float3 direction, float minT, float maxT)
{
    // Setup our reflection ray
    RayDesc ray;
    ray.Origin = origin;
    ray.Direction = direction;
    ray.TMin = minT;
    ray.TMax = maxT;

    // Initialize the ray's payload data with black return color
    GGXPayload rayPayload;
    rayPayload.color = float3(0, 0, 0);

    TraceRay(gScene.rtAccel, RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH, 0xFF, 0, rayTypeCount, 0, ray, rayPayload);

    return rayPayload.color;
}

[shader("miss")]
void GGXMiss(inout GGXPayload rayData)
{
}

[shader("anyhit")]
void GGXAnyHit(inout GGXPayload rayData, BuiltInTriangleIntersectionAttributes attribs)
{
    // Run a Falcor helper to extract the current hit point's geometric data
    GeometryInstanceID instanceID = getGeometryInstanceID();
    VertexData v = getVertexData(instanceID, PrimitiveIndex(), attribs);
    const uint materialID = gScene.getMaterialID(instanceID);

    // Test if this hit point passes a standard alpha test.  If not, discard/ignore the hit.
    if (alphaTest(v, gScene.materials[materialID], gScene.materialResources[materialID], 0.f))
        IgnoreHit();
}

[shader("closesthit")]
void GGXClosestHit(inout GGXPayload rayData, BuiltInTriangleIntersectionAttributes attribs)
{
    // Get the hit-point data
    const GeometryInstanceID instanceID = getGeometryInstanceID();
    VertexData v = getVertexData(instanceID, PrimitiveIndex(), attribs);
    uint materialID = gScene.getMaterialID(instanceID);

    // Extract Falcor scene data for shading
    ShadingData shadeData = prepareShadingData(v, materialID, gScene.materials[materialID], gScene.materialResources[materialID], -WorldRayDirection(), 0);

    //// Extract Falcor scene data for shading
    //ShadingData shadeData = getHitShadingData(hitParams, attribs, WorldRayOrigin());

    /*const uint lightCount = gScene.getLightCount();
    int shadowMult = lightCount;
    for (int lightIndex = 0; lightIndex < lightCount; lightIndex++){
        float distToLight;
        float3 lightIntensity;
        float3 L;
        // A helper (that queries the Falcor scene to get needed data about this light)
        getLightData(lightIndex, shadeData.posW, L, lightIntensity, distToLight);
        if (gVisibility[shadeData.posW.xy] & (1 << lightIndex)) shadowMult--;
    }
    if (shadowMult < lightCount) rayData.color = float3(0.7f) * shadeData.specular.rgb;
    else rayData.color = float3(0.0f);*/
    rayData.color = shadeData.specular.rgb;
}


[shader("raygeneration")]
void GGXRayGen()
{
    // Where is this ray on screen?
    uint2 launchIndex = DispatchRaysIndex().xy;
    uint2 launchDim = DispatchRaysDimensions().xy;

    // Load g-buffer data
    float4 worldPos = gPos[launchIndex];
    float4 worldNorm = gNorm[launchIndex];
    //float4 VColor = gVshading[launchIndex];
    float  visibility = gVisibility[launchIndex];
    uint raymask = gRaymask[launchIndex];
    // Get the texture data that is stored in a compact format
    float4 difMatlColor;
    float4 specMatlColor;
    float4 pixelEmissive;
    float4 matlOthers;
    unpackTextureData(asuint(gTexData[launchIndex]), difMatlColor, specMatlColor, pixelEmissive, matlOthers);

    if (dot(difMatlColor.rgb, difMatlColor.rgb) < 0.00001f) difMatlColor = specMatlColor;

    // Does this g-buffer pixel contain a valid piece of geometry?  (0 in pos.w for invalid)
    bool isGeometryValid = (worldPos.w != 0.0f);

    // Extract and compute some material and geometric parameters. 
    //float roughness = specMatlColor.a * specMatlColor.a;
    //float reflectable = ggxNormalDistribution()

    // If we don't hit any geometry, our difuse material contains our background color.
    float3 shadeColor = float3(0, 0, 0);

    /* We skip based on the following:
    *  1. We are skipping SRT
    *  2. Surface is not shiny enough (represented by large roughness)
    */      
    if (gSkipSRT || (gUseThresholds && (specMatlColor.a * specMatlColor.a > gRoughnessThreshold)))
    {
        gOutput[launchIndex] = float3(0.0);
        return;
    }
    
    if (isGeometryValid && !raymask)
    {
        //shadeColor = VColor.rgb + pixelEmissive.rgb;

        const uint lightCount = gScene.getLightCount();
        for (int lightIndex = 0; lightIndex < lightCount; lightIndex++)
        {
            float distToLight;
            float3 lightIntensity;
            float3 L;

            float shadowMult = (visibility & (1 << lightIndex)) ? 1.0 : 0.0;

            if (shadowMult == 1.0)
            {
                // A helper (that queries the Falcor scene to get needed data about this light)
                getLightData(lightIndex, worldPos.xyz, L, lightIntensity, distToLight);
                
                // Get mirror reflection vector
                float3 V = normalize(gScene.camera.getPosition() - worldPos.xyz);
                float3 H = normalize(V + L);
                float3 R = normalize(reflect(V, worldNorm.xyz));
                float roughness = specMatlColor.a * specMatlColor.a;
                float NdotL = saturate(dot(worldNorm.xyz, L));
                float NdotH = saturate(dot(worldNorm.xyz, H));
                float LdotH = saturate(dot(L, H));
                float NdotV = saturate(dot(worldNorm.xyz, V));
                bool valid = NdotV * NdotL * LdotH > 0.0f;

                if (valid)
                {
                    // Shoot secondary ray
                    float3 GGXcolor = shootGGXray(worldPos.xyz, R, gMinT, 1.0e38f);
                    //float3 GGXcolor = gSkipGGX ? float3(0, 0, 0) : float3(0, 0, 0.3);
                    float  D = ggxNormalDistribution(NdotH, roughness);
                    float  G = ggxSchlickMaskingTerm(NdotL, NdotV, roughness);
                    float3 F = schlickFresnel(specMatlColor.rgb, LdotH);

                    float3 BRDF = F * G * D / (4.f * NdotV * NdotL);

                    shadeColor += GGXcolor * float3(BRDF);
                }
            }
        }
    }
    
    // Clamp function doesn't seem to work for some reason. So this is a workaround
    shadeColor.r = max(0.0, min(shadeColor.r, 1.0));
    shadeColor.g = max(0.0, min(shadeColor.g, 1.0));
    shadeColor.b = max(0.0, min(shadeColor.b, 1.0));
    
    if (gUseThresholds && luminance(shadeColor) < gLumThreshold)
    {
        gOutput[launchIndex] = float3(0);
        return;
    }
    
    //if(gRaymask[launchIndex] == 0) gOutput[launchIndex] = float4(0, 1, 0, 1.0f);
    // else if(gRaymask[launchIndex] == 1) gOutput[launchIndex] = float4(1, 0, 0, 1.0f);
    gOutput[launchIndex] = shadeColor;
}