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
import Scene.Shading;                      // Shading functions, etc     

// A separate file with some simple utility functions: getPerpendicularVector(), initRand(), nextRand()
#include "aoCommonUtils.hlsli"

// Payload for our primary rays. 
struct AORayPayload
{
	float hitDist;
    float3 hitColor;
};

// A constant buffer we'll fill in for our ray generation shader
cbuffer RayGenCB
{
    bool  gSkipAo;
	float gAORadius;
	uint  gFrameCount;
	float gMinT;
	uint  gNumRays;
}

// Input and out textures that need to be set by the C++ code
Texture2D<float4> gPos;
Texture2D<float4> gNorm;
RWTexture2D<uint4> gOutput;


[shader("miss")]
void AoMiss(inout AORayPayload hitData : SV_RayPayload)
{
}

[shader("anyhit")]
void AoAnyHit(inout AORayPayload rayData, BuiltInTriangleIntersectionAttributes attribs)
{
	// Is this a transparent part of the surface?  If so, ignore this hit
	if (alphaTestFails(attribs))
		IgnoreHit();
}

[shader("closesthit")]
void AoClosestHit(inout AORayPayload rayData, BuiltInTriangleIntersectionAttributes attribs)
{
	// We update the hit distance with our current hitpoint
    rayData.hitDist = RayTCurrent();
	
	// Get the hit-point data
    const GeometryInstanceID instanceID = getGeometryInstanceID();
    VertexData v = getVertexData(instanceID, PrimitiveIndex(), attribs);
    uint materialID = gScene.getMaterialID(instanceID);
	
	 // Extract Falcor scene data for shading
    ShadingData shadeData = prepareShadingData(v, materialID, gScene.materials[materialID], gScene.materialResources[materialID], -WorldRayDirection(), 0);
	
    rayData.hitColor = shadeData.diffuse.rgb;
}


[shader("raygeneration")]
void AoRayGen()
{
	// Where is this ray on screen?
	uint2 launchIndex = DispatchRaysIndex().xy;
	uint2 launchDim   = DispatchRaysDimensions().xy;

	// Initialize random seed per sample based on a screen position and temporally varying count
	uint randSeed = initRand(launchIndex.x + launchIndex.y * launchDim.x, gFrameCount, 16);

	// Load the position and normal from our g-buffer
	float4 worldPos = gPos[launchIndex];
	float4 worldNorm = gNorm[launchIndex];

	// Default ambient occlusion
	uint ambientOcclusion = gNumRays;
    float3 accumGlobalIllum = float3(0.0f);
	
	// Our camera sees the background if worldPos.w is 0, only shoot an AO ray elsewhere
	if (worldPos.w != 0.0f)  
	{
        if (gSkipAo)
        {
			// Skipping AO, so mark all rays as unoccluded.
            gOutput[launchIndex] = gNumRays;
            return;
        }
		
		// Start accumulating from zero if we don't hit the background
		ambientOcclusion = 0;

		for (int i = 0; i < gNumRays; i++)
		{
			// Sample cosine-weighted hemisphere around the surface normal
			float3 worldDir = getCosHemisphereSample(randSeed, worldNorm.xyz);

			// Setup ambient occlusion ray
			RayDesc rayAO;
			rayAO.Origin = worldPos.xyz;
			rayAO.Direction = worldDir;
			rayAO.TMin = gMinT;
            rayAO.TMax = 1.0e38f;

			// Initialize the maximum hitT (which will be the return value if we hit no surfaces)
            AORayPayload rayPayload = { gAORadius + 1.0f, float3(0.0f) };

			// Trace our ray
            uint rayFlags = RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH;
			
            TraceRay(gScene.rtAccel, rayFlags, 0xFF, 0, rayTypeCount, 0, rayAO, rayPayload);
			
			// If our hit is what we initialized it to, above, we hit no geometry (else we did hit a surface)
            if (rayPayload.hitDist > gAORadius)
            {
                ambientOcclusion += 1;
                //accumGlobalIllum += rayPayload.hitColor;
            }
			
		}
	}
	
	// Save out our AO color
 //   uint2 actualIndex = uint2(launchIndex.x, launchIndex.y >> 2); // We use a quarter of the actual texture size.
 //   // Every 32-bits has the following format
	////			 |   8   |   8   |   8   |   8   |
	////   y % 4   |   0   |   1   |   2   |   3   |
 //   uint shiftFactor = 24 - 8 * (launchIndex.y % 4);
 //   gOutput[actualIndex] = (gOutput[actualIndex] | ((ambientOcclusion) << shiftFactor));
    gOutput[launchIndex] = ambientOcclusion;
}
