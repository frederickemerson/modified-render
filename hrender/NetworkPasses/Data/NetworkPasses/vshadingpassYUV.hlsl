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
    bool gSkipShadows; // Render all lights without shadow rays
    bool gSkipAO; // Render the scene without ambient occlusion
    bool gDecodeMode; // Just debug the visibility bitmaps
    int gDecodeBit; // Which light of the visibility bitmap to preview
    float gAmbient;
    uint gNumAORays;
}

// Input and out textures that need to be set by the C++ code
Texture2D<float4> gPos;
Texture2D<float4> gNorm;
Texture2D<float4> gTexData;
Texture2D<uint>  gVisibility;
Texture2D<uint> gAO;
RWTexture2D<uint> gOutput;

// Values used for RGB -> YUV conversion. Alpha value ignored.
//uint gHeight;  // Height of the output image (1080 currently)
static float3x3 rgbToYuv = { 0.299, -0.169,  0.499,
                             0.587, -0.331, -0.418,
                             0.114, 0.499, -0.0813 };
static float3 offset = { 0.0, 128.0, 128.0 }; 

[shader("raygeneration")]
void VShadowsRayGen()
{
    // Where is this ray on screen?
    uint2 launchIndex = DispatchRaysIndex().xy;
    uint2 launchDim = DispatchRaysDimensions().xy;

    // Load g-buffer data
    float4 worldPos = gPos[launchIndex];
    float4 worldNorm = gNorm[launchIndex];
    float4 difMatlColor = unpackUnorm4x8(asuint(gTexData[launchIndex].x));

    // We're only doing Lambertian, but sometimes Falcor gives a black Lambertian color.
    //    There, this shader uses the spec color for our Lambertian color.
    float4 specMatlColor = unpackUnorm4x8(asuint(gTexData[launchIndex].y));
    if (dot(difMatlColor.rgb, difMatlColor.rgb) < 0.00001f) difMatlColor = specMatlColor;

    // If we don't hit any geometry, our difuse material contains our background color.
    float3 shadeColor = difMatlColor.rgb;


    if (!gDecodeMode)
    {
        // Our camera sees the background if worldPos.w is 0, only shoot an AO ray elsewhere
        if (worldPos.w != 0.0f)
        {
            // We're going to accumulate contributions from multiple lights, so zero our our sum
            shadeColor = float3(0.0, 0.0, 0.0);

            const uint lightCount = gScene.getLightCount();
            float3 lightIntensityCache;

            for (int lightIndex = 0; lightIndex < lightCount; lightIndex++)
            {
                float distToLight;
                float3 lightIntensity;
                float3 toLight;
                // A helper (that queries the Falcor scene to get needed data about this light)
                getLightData(lightIndex, worldPos.xyz, toLight, lightIntensity, distToLight);

                // Compute our lambertion term (L dot N)
                float LdotN = saturate(dot(worldNorm.xyz, toLight));

                // Shoot our ray
                float shadowMult = gSkipShadows ? 1.0f :
                    ((gVisibility[launchIndex] & (1 << lightIndex)) ? 1.0 : 0.0);
                lightIntensityCache = lightIntensity;
                // Compute our Lambertian shading color
                shadeColor += shadowMult * LdotN * lightIntensity;
            }
            if (length(shadeColor) < 0.00001)
            {
                shadeColor += difMatlColor.rgb * gAmbient * lightIntensityCache;
            }

            // Physically based Lambertian term is albedo/pi
            shadeColor *= difMatlColor.rgb / 3.141592f;
        }
        // Testing color
        float3 yuvOutput = mul(shadeColor * 255, rgbToYuv);
        yuvOutput += offset + float3(0.5, 0.5, 0.5);
        yuvOutput = clamp(trunc(yuvOutput), float3(0.0, 0.0, 0.0), float3(255.0, 255.0, 255.0));

        // Save out our AO color
        // 32 bit int : Y (8bits) | U (8 bits) | V (8bits) | unused (8bits)
        uint finalColor = (uint) yuvOutput.r << 24 | (uint) yuvOutput.g << 16 | (uint) yuvOutput.b << 8;
        gOutput[launchIndex] = finalColor;
    }
    else
    {
        float shadowMult = gSkipShadows ? 1.0f :
            ((gVisibility[launchIndex] & (1 << gDecodeBit)) ? 1.0 : 0.0f);
        
        shadeColor *= shadowMult;
        
        float3 yuvOutput = mul(shadeColor * 255, rgbToYuv);
        yuvOutput += offset + float3(0.5, 0.5, 0.5);
        yuvOutput = clamp(trunc(yuvOutput), float3(0.0, 0.0, 0.0), float3(255.0, 255.0, 255.0));
        // Save out our AO color
        uint finalColor = (uint) yuvOutput.r << 24 | (uint) yuvOutput.g << 16 | (uint) yuvOutput.b << 8;
        gOutput[launchIndex] = finalColor;
    }

}
