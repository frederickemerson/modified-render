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

#include "packingUtils.hlsli"
#include "Utils/Math/MathConstants.slangh"

// Include and import common Falcor utilities and data structures
import Scene.Raytracing;
import Scene.Shading;                      // Shading functions, etc   
import Scene.Scene;
import Scene.Camera.Camera;

// Include utility functions for sampling random numbers
#include "lightProbeGBufferUtils.hlsli"

// Payload for our primary rays.  We really don't use this for this g-buffer pass
struct SimpleRayPayload
{
    bool hit;
};

// A constant buffer we'll fill in for our miss shader
cbuffer MissShaderCB
{
    uint2   gEnvMapRes;
};

// Our texture containing our environment map
Texture2D<float4>   gEnvMap;

// Shader parameters for our ray gen shader that need to be set by the C++ code
cbuffer RayGenCB
{
    float   gLensRadius;
    float   gFocalLen;
    uint    gFrameCount;
    bool    gUseThinLens;
    float2  gPixelJitter;   // in [0..1]^2.  Should be (0.5,0.5) if no jittering used
}

// Our output textures, where we store our G-buffer results
shared RWTexture2D<float4> gWsPos;
shared RWTexture2D<float4> gWsNorm;
// This render target stores material texture data in a packed format with 8 bits per component.
// r: diffuse.r,    diffuse.g,      diffuse.b,          opacity
// g: specular.r,   specular.g,     specular.b,         linear roughness
// b: emissive.r,   emissive.g,     emissive.b,         doubleSided ? 1.0f : 0.0f
// a: IoR,          metallic,       specular trans,     eta
shared RWTexture2D<float4> gTexData;

[shader("miss")]
void PrimaryMiss(inout SimpleRayPayload hitData)
{
    // Where do we store our environment color (i.e., which pixel missed the geometry?)
    uint2 launchIndex = DispatchRaysIndex().xy;

    // Convert our direction to a (u,v) coordinate
    float2 uv = wsVectorToLatLong(WorldRayDirection());

    // Lookup and return our light probe color
    gTexData[launchIndex] = float4(
        asfloat(packUnorm4x8(float4(gEnvMap[uint2(uv * gEnvMapRes)].rgb, 1.0f))),
        0.f, 0.f, 0.f
    );
}

[shader("anyhit")]
void PrimaryAnyHit(uniform HitShaderParams hitParams, inout SimpleRayPayload hitData, BuiltInTriangleIntersectionAttributes attribs)
{
    // Run a Falcor helper to extract the current hit point's geometric data
    VertexData v = getVertexData(hitParams, PrimitiveIndex(), attribs);
    const uint materialID = gScene.getMaterialID(hitParams.getGlobalHitID());

    // Test if this hit point passes a standard alpha test.  If not, discard/ignore the hit.
    if (alphaTest(v, gScene.materials[materialID], gScene.materialResources[materialID], 0.f))
        IgnoreHit();
}

[shader("closesthit")]
void PrimaryClosestHit(uniform HitShaderParams hitParams, inout SimpleRayPayload rayData, BuiltInTriangleIntersectionAttributes attribs)
{
    rayData.hit = true;

    // Get some information about the current ray
    uint2  launchIndex     = DispatchRaysIndex().xy;  
    float3 cameraPosW = gScene.camera.getPosition();
    float3 rayDirW = WorldRayDirection();
    uint materialID = gScene.getMaterialID(hitParams.getGlobalHitID());

    // Run a pair of Falcor helper functions to compute important data at the current hit point
    VertexData v = getVertexData(hitParams, PrimitiveIndex(), attribs);
    ShadingData shadeData = prepareShadingData(v, materialID, gScene.materials[materialID], gScene.materialResources[materialID], -rayDirW, 0);

    // Save out our G-Buffer values to our textures
    gWsPos[launchIndex]    = float4(shadeData.posW, 1.f);
    gWsNorm[launchIndex]   = float4(shadeData.N, length(shadeData.posW - cameraPosW));
    gTexData[launchIndex]  = asfloat(packTextureData(shadeData));
}


[shader("raygeneration")]
void GBufferRayGen()
{
    Camera camera = gScene.camera;

    // Get our pixel's position on the screen
    uint2 launchIndex = DispatchRaysIndex().xy;
    uint2 launchDim   = DispatchRaysDimensions().xy;

    // Convert our ray index into a ray direction in world space
    float2 pixelCenter = (launchIndex + gPixelJitter) / launchDim;
    float2 ndc = float2(2, -2) * pixelCenter + float2(-1, 1);                    
    float3 rayDir = ndc.x * camera.data.cameraU + ndc.y * camera.data.cameraV + camera.data.cameraW;  
    rayDir /= length(camera.data.cameraW);

    // Find the focal point for this pixel.
    float3 focalPoint = camera.data.posW + gFocalLen * rayDir;

    // Initialize a random number generator
    uint randSeed = initRand(launchIndex.x + launchIndex.y * launchDim.x, gFrameCount, 16);

    // Get random numbers (polar coordinates), convert to random cartesian uv on the lens
    float2 rnd = float2(2.0f * M_PI * nextRand(randSeed), gLensRadius * nextRand(randSeed));
    float2 uv  = float2(cos(rnd.x) * rnd.y, sin(rnd.x) * rnd.y);

    // Use uv coordinate to compute a random origin on the camera lens
    float3 randomOrig = camera.data.posW + uv.x * normalize(camera.data.cameraU) + uv.y * normalize(camera.data.cameraV);

    // Initialize a ray structure for our ray tracer
    RayDesc ray; 
    ray.Origin    = gUseThinLens ? randomOrig : camera.data.posW;  
    ray.Direction = normalize(gUseThinLens ? focalPoint - randomOrig : rayDir);
    ray.TMin      = 0.0f;              
    ray.TMax      = 1e+38f;            

    // Initialize our ray payload (a per-ray, user-definable structure)
    SimpleRayPayload rayData = { false };

    // Trace our ray
    TraceRay(gRtScene,                        // Acceleration structure
        RAY_FLAG_CULL_BACK_FACING_TRIANGLES,  // Ray flags
        0xFF,                                 // Instance inclusion mask
        0,                                    // Ray type
        hitProgramCount,                      // 
        0,                                    // Miss program index
        ray,                                  // Ray to shoot
        rayData);                             // Our ray payload
}
