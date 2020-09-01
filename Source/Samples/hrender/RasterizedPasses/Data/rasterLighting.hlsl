// Include and import common Falcor utilities and data structures
import Scene.Raytracing;
import Scene.Shading;                      // Shading functions, etc   
import Scene.Lights.Lights;                // Light structures for our current scene

#include "Utils/Math/MathConstants.slangh"
#include "../../DxrTutorCommonPasses/Data/CommonPasses/packingUtils.hlsli"  // Functions used to unpack the GBuffer's gTexData
#include "../../DxrTutorCommonPasses/Data/CommonPasses/lambertianPlusShadowsUtils.hlsli"
#include "shadowmapCommon.hlsli"

// Input from GBuffer
Texture2D<float4>           gPos;
Texture2D<float4>           gNorm;
Texture2D<float4>           gTexData;
// Texture2D[6] which holds the shadow map (depth^2 view from the light's perspective).
// It stores values from [0, 1] - distance^2(posW,lightPosW) / gShadowFar2 
Texture2DArray<float>       gShadowMap;
// Used to store the point light posW's, to draw them for debugging
StructuredBuffer<float3>    lightLocations;

cbuffer RasterBuf
{
    // Constants used related to shadow map shading
    uint    gShadowMapRes;          // Resolution of the shadow map used
    float   gShadowFar2;            // Far plane distance of the shadow map squared
    float   gShadowOffset;          // Distance to offset shadow map distances to prevent shadow acne
    // Per-light variables for this shadow map shading pass (this pass is executed once per light)
    uint    gCurrLightIdx;          // Light index currently being rendered 
    bool    gUsingShadowMapping;    // Whether this light uses shadow mapping
    // Constants used for debug light drawing
    bool    gShowLights;            // Whether or not debug lights are being drawn
    uint    gNumPointLights;        // Number of point lights in the scene, used to draw them for debugging
    float   gLightRadius;           // The radius to draw the lights for debugging
};

// Color output from this shader 
RWTexture2D<float4>         gOutput;

// Returns 1 if a point is visible to the light, 0 otherwise
float shadowMapVisibility(float3 posW)
{
    // TODO: Direction lights don't have shadow map support yet
    if (!gUsingShadowMapping) return 1.0f;

    // Vector from the light to the fragment's location
    float3 viewVec = posW - gScene.getLight(gCurrLightIdx).posW;
    // Get the actual depth value squared. We use square to prevent having to use sqrt.
    float actualDepth2 = dot(viewVec, viewVec);
    // Normalize this to [0, 1]
    float distNormalized = actualDepth2 / gShadowFar2;
    // Get the depth value in the shadow map
    float3 uvi = dirToCubeCoords(normalize(viewVec));
    float shadowMapDepth = gShadowMap[uvi * uint3(gShadowMapRes, gShadowMapRes, 1)];

    // Return true if the actual depth is what was seen by light
    return distNormalized <= shadowMapDepth + gShadowOffset;
}

[shader("raygeneration")]
void RasterizationRayGen()
{
    // Where is this ray on screen?
    uint2 launchIndex = DispatchRaysIndex().xy;
    uint2 launchDim   = DispatchRaysDimensions().xy;

    // Load g-buffer data
    float4 worldPos     = gPos[launchIndex];
    float4 worldNorm    = gNorm[launchIndex];
    float4 difMatlColor = unpackUnorm4x8(asuint(gTexData[launchIndex].x));

    // Preview the light positions in the scene
    if (gShowLights)
    {
        for (int lightIndex = 0; lightIndex < gNumPointLights; lightIndex++)
        {
            // If this pixel location is close enough to the light's pixel location, show a dot to represent the light.
            if (distance(lightLocations[lightIndex].xy * launchDim, launchIndex) < gLightRadius)
            {
                gOutput[launchIndex] = float4(frac(lightLocations[lightIndex]), 1.0f);
                return;
            }
        }
    }

    // We're only doing Lambertian, but sometimes Falcor gives a black Lambertian color.
    //    There, this shader uses the spec color for our Lambertian color.
    float4 specMatlColor = unpackUnorm4x8(asuint(gTexData[launchIndex].y));
    if (dot(difMatlColor.rgb, difMatlColor.rgb) < 0.00001f) difMatlColor = specMatlColor;

    // If we don't hit any geometry, our difuse material contains our background color.
    float3 shadeColor = difMatlColor.rgb;

    // Our camera sees the background if worldPos.w is 0, only shoot an AO ray elsewhere
    if (worldPos.w != 0.0f)
    {
        // We're going to accumulate contributions from multiple lights, so zero our our sum
        shadeColor = float3(0.0, 0.0, 0.0);

        float distToLight;
        float3 lightIntensity;
        float3 toLight;
        // A helper (that queries the Falcor scene to get needed data about this light)
        getLightData(gCurrLightIdx, worldPos.xyz, toLight, lightIntensity, distToLight);

        // Compute our lambertion term (L dot N)
        float LdotN = saturate(dot(worldNorm.xyz, toLight));

        // Perform visibility check
        float shadowMult = shadowMapVisibility(worldPos.xyz);

        // Compute our Lambertian shading color
        shadeColor += shadowMult * LdotN * lightIntensity; 

        // Physically based Lambertian term is albedo/pi
        shadeColor *= difMatlColor.rgb / M_PI;
    }
   
    // Save out our AO color. If we hit actual scene geometry, add the color contribution, otherwise
    // it's the environment map and we just write the color contribution directly.
    gOutput[launchIndex] = (worldPos.w != 0.0f)
        ? saturate(gOutput[launchIndex] + float4(shadeColor, 1.0f))
        : float4(shadeColor, 1.0f);
}
