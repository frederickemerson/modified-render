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
import Experimental.Scene.Lights.LightHelpers; // Light structures for our current scene

#include "../../../DxrTutorCommonPasses/Data/CommonPasses/packingUtils.hlsli" // Functions used to unpack the GBuffer's gTexData

Texture2D<float4> gPos;
Texture2D<float4> gNorm;
Texture2D<float4> gTexData;
Texture2D<uint>   gVisibility;
Texture2D<float>  gZBuffer;

cbuffer SSRCB
{
	bool     gSkipSSR;
	float4x4 gViewProjMat;
	uint     gLightCount;
	float3   gCamPos;
}

struct PS_OUTPUT
{
	uint   RayMask   : SV_Target0;
	float4 SSRColor  : SV_Target1;
};

float3 SSRRayMarch(float3 origin, float3 direction)
{
	int iteration = 50;
	float3 temp = origin;
	for (int i = 0; i < iteration; i++) {
		// Test point goes along the direction
		origin += direction * 3;

		// World position to screen position
		float4 posS = mul(float4(origin, 1.0), gViewProjMat);
		posS.xyz /= posS.w;
		posS.xyz = posS.xyz * 0.5f + 0.5f;
		if (posS.x < 0 || posS.x > 1 || posS.y < 0 || posS.y > 1) {
			return float3(0.0f);
		}
		float depth = gZBuffer[posS.xy];
		if (depth < posS.z ) {
			float4 difMatlColor;
			float4 specMatlColor;
			float4 pixelEmissive;
			float4 matlOthers;
			unpackTextureData(asuint(gTexData[origin.xy]), difMatlColor, specMatlColor, pixelEmissive, matlOthers);
			//return  2 * gVshading[origin.xy].rgb;
			return  specMatlColor.rgb;
		}
	}

	return float3(0.0f);
}

PS_OUTPUT main(float2 texC : TEXCOORD, float4 pos : SV_Position)
{
	PS_OUTPUT SSRBufOut;
	uint2 pixelPos = (uint2)pos.xy;

	// Load g-buffer data
	float4 worldPos = gPos[pixelPos];
	float4 worldNorm = gNorm[pixelPos];
	float  visibility = gVisibility[pixelPos];
	// Get the texture data that is stored in a compact format
	float4 difMatlColor;
	float4 specMatlColor;
	float4 pixelEmissive;
	float4 matlOthers;
	unpackTextureData(asuint(gTexData[pixelPos]), difMatlColor, specMatlColor, pixelEmissive, matlOthers);

	// Does this g-buffer pixel contain a valid piece of geometry?  (0 in pos.w for invalid)
	bool isGeometryValid = (worldPos.w != 0.0f);
	// If we don't hit any geometry, our difuse material contains our background color.
	float3 shadeColor = isGeometryValid ? float3(0.0f) : difMatlColor.rgb;
	// If geometry invalid or we skip SSR, this pixel is discard for raytracing.
	SSRBufOut.RayMask = (!isGeometryValid || gSkipSSR) ? 1 : 0;
	// We use this temp color to check SSR hit or miss
	float3 SSRColor = float3(0);
	float roughness = specMatlColor.a * specMatlColor.a;

	if (isGeometryValid)
	{

		// Set V-shading color and emissive color.
		shadeColor = float3(0);

		for (int lightIndex = 0; lightIndex < gLightCount; lightIndex++)
		{
			float shadowMult = (visibility & (1 << lightIndex)) ? 1.0 : 0.0;

			if (shadowMult == 1.0)
			{
				float3 V = normalize(gCamPos - worldPos.xyz);
				float3 R = normalize(reflect(V, worldNorm.xyz));

				// Do SSR
				float3 ssrColor = gSkipSSR ? float3(0, 0, 0) : SSRRayMarch(worldPos.xyz, R);

				if (SSRColor.r != 0.0f && SSRColor.g != 0.0f && SSRColor.b != 0.0f) SSRBufOut.RayMask = 1;
				// roughness helps reduce noise
				SSRColor += roughness * ssrColor;

				
			}
		}

	}

	// If shaderColor has been changed, means SSR worked for this pixel, no need to do raytracing.
	if (SSRBufOut.RayMask == 1) shadeColor =  SSRColor;

	if (shadeColor.r >= 1.0f)shadeColor.r = 1.0f;
	else if (shadeColor.g >= 1.0f)shadeColor.g = 1.0f;
	else if (shadeColor.b >= 1.0f)shadeColor.b = 1.0f;

	SSRBufOut.SSRColor = float4(shadeColor, 1.0f);

	return SSRBufOut;
}