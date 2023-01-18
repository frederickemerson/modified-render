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

#include "../../../DxrTutorCommonPasses/Data/CommonPasses/packingUtils.hlsli"              // Functions used to unpack the GBuffer's gTexData
// A separate file with some simple utility functions: getPerpendicularVector(), initRand(), nextRand()
#include "networkUtils.hlsli"

// Payload for our primary rays.  We really don't use this for this g-buffer pass
struct ShadowRayPayload
{
    int hitDist;
};

// A constant buffer we'll fill in for our ray generation shader
cbuffer RayGenCB
{
    float gMinT;
    uint gFrameCount;
    bool gUseConeSampling; // True if cone sampling is to be used.
    float gCosThetaMax; // Has a range of [0, pi/2]
    bool gSkipShadows; // Render all lights without shadow rays
}

// Input and out textures that need to be set by the C++ code
Texture2D<float4> gPos;
RWTexture2D<uint> gOutput;

// Cone sampling for direct shadows
float3 uniformSampleCone(inout uint randSeed, float cosThetaMax)
{
    // Get 2 random numbers to select our sample with
    float2 randVal = float2(nextRand(randSeed), nextRand(randSeed));
    float cosTheta = (1.0 - randVal.x) + randVal.x * cosThetaMax;
    float sinTheta = sqrt(1 - cosTheta * cosTheta);
    float phi = 2.0f * 3.14159265f * randVal.y;
   
    return normalize(float3(cos(phi) * sinTheta, sin(phi) * sinTheta, cosTheta));
}

float shadowRayVisibility( float3 origin, float3 direction, float minT, float maxT )
{
    // Setup our shadow ray
    RayDesc ray;
    ray.Origin = origin;
    ray.Direction = direction;
    ray.TMin = minT;
    ray.TMax = maxT; 

    // Query if anything is between the current point and the light (i.e., at maxT) 
    ShadowRayPayload rayPayload = { maxT + 1.0f }; 
    TraceRay(gScene.rtAccel, RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH, 0xFF, 0, rayTypeCount, 0, ray, rayPayload);

    // Check if anyone was closer than our maxT distance (in which case we're occluded)
    return (rayPayload.hitDist > maxT) ? 1.0f : 0.0f;
}

[shader("miss")]
void ShadowMiss(inout ShadowRayPayload rayData)
{
}

[shader("anyhit")]
void ShadowAnyHit(inout ShadowRayPayload rayData, BuiltInTriangleIntersectionAttributes attribs)
{
    // Run a Falcor helper to extract the current hit point's geometric data
    GeometryInstanceID instanceID = getGeometryInstanceID();
    VertexData v = getVertexData(instanceID, PrimitiveIndex(), attribs);
    const uint materialID = gScene.getMaterialID(instanceID);

    // Test if this hit point passes a standard alpha test.  If not, discard/ignore the hit.
    if (alphaTest(v, gScene.materials[materialID], gScene.materialResources[materialID], 0.f))
        IgnoreHit();

    // We update the hit distance with our current hitpoint
    rayData.hitDist = RayTCurrent();
}

[shader("closesthit")]
void ShadowClosestHit(inout ShadowRayPayload rayData, BuiltInTriangleIntersectionAttributes attribs)
{
    rayData.hitDist = RayTCurrent();
}


[shader("raygeneration")]
void SimpleShadowsRayGen()
{
    // Where is this ray on screen?
    uint2 launchIndex = DispatchRaysIndex().xy;
    uint2 launchDim   = DispatchRaysDimensions().xy;
    
    // Initialize random seed per sample based on a screen position and temporally varying count
    uint randSeed = initRand(launchIndex.x + launchIndex.y * launchDim.x, gFrameCount, 16);
    
    // Load g-buffer data
    float4 worldPos     = gPos[launchIndex];

    // If we don't hit any geometry, our difuse material contains our background color.
    int visibilityBit = 0;

    // Our camera sees the background if worldPos.w is 0, only shoot an AO ray elsewhere
    if (worldPos.w != 0.0f)
    {

        const uint lightCount = gScene.getLightCount();
        for (int lightIndex = 0; lightIndex < lightCount; lightIndex++)
        {
            float distToLight;
            float3 lightIntensity;
            float3 toLight;
            // A helper (that queries the Falcor scene to get needed data about this light)
            getLightData(lightIndex, worldPos.xyz, toLight, lightIntensity, distToLight);

            if (gUseConeSampling)
            {
                float3 rndDirection = uniformSampleCone(randSeed, gCosThetaMax);
                
                // Getting the direction in terms of the basis
                float3 bitangent = getPerpendicularVector(toLight);
                float3 tangent = cross(bitangent, toLight);
                
                toLight = tangent * rndDirection.x + bitangent * rndDirection.y + toLight * rndDirection.z;
                toLight = normalize(toLight);
                distToLight = distToLight / gCosThetaMax;
            }
            
            // Shoot our ray
            float visibility = gSkipShadows ? 1.0f : shadowRayVisibility(worldPos.xyz, toLight, gMinT, distToLight);

            // Store visibility
            visibilityBit |= (int(visibility) << lightIndex);
        }
    }
    
    // Save out our AO color
    gOutput[launchIndex] = visibilityBit;
}
