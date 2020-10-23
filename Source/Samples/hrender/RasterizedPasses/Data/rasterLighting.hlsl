// Include and import common Falcor utilities and data structures
import Scene.Raytracing;
import Scene.Shading;                      // Shading functions, etc   
import Scene.Lights.Lights;                // Light structures for our current scene

#include "Utils/Math/MathConstants.slangh"
#include "../../DxrTutorCommonPasses/Data/CommonPasses/packingUtils.hlsli"  // Functions used to unpack the GBuffer's gTexData
#include "../../DxrTutorCommonPasses/Data/CommonPasses/lambertianPlusShadowsUtils.hlsli"
#include "shadowmapCommon.hlsli"

cbuffer RasterBuf
{
    // Constants used related to shadow map shading
    bool        gUsingShadowMapping;    // Whether this pass uses shadow mapping
    uint        gCubeShadowMapRes;      // Resolution of the cube shadow map used
    uint        gDirShadowMapRes;       // Resolution of the directional shadow map used
    float       gCubeShadowFar2;        // Far plane distance of the shadow map squared
    float       gCubeShadowBias;        // Distance to offset cube shadow map distances to prevent shadow acne
    float       gDirShadowBias;         // Distance to offset dir shadow map distances to prevent shadow acne
    float3      gDirOrigin;             // Origin of the directional shadow map
    float4x4    gDirViewProj;           // View-Projection matrix of directional shadow map
    // Constants used for debug light drawing
    bool        gShowLights;            // Whether or not debug lights are being drawn
    uint        gNumPointLights;        // Number of point lights in the scene, used to draw them for debugging
    float       gLightRadius;           // The radius to draw the lights for debugging
    // Constants used for percentage-closer filtering (AA)
    bool        gUsingPCF;              // Whether or not we are using PCF
    float       gCubePCFWidth;          // The width of the cube PCF filter
    float       gDirPCFWidth;           // The width of the dir PCF filter
};

// Used to store the point light posW's, to draw them for debugging
StructuredBuffer<float3>    gLightLocations;

// Input from GBuffer
Texture2D<float4>           gPos;
Texture2D<float4>           gNorm;
Texture2D<float4>           gTexData;
// Texture that holds the directional shadow map
Texture2D<float>            gDirShadowMap;
// Texture2D[numPointLights] which holds the shadow map (depth^2 view from the light's perspective).
// It stores values from [0, 1] - distance^2(posW,lightPosW) / gCubeShadowFar2 
Texture2DArray<float>       gCubeShadowMap;

// Color output from this shader 
RWTexture2D<float4>         gOutput;

// Returns 1 if a point is visible to the light, 0 otherwise.
// Directional lights always return 1 for now.
// pointLightIndex will be the offset / 6 of the cube map arrays to access if the light
// is a point light, -1 otherwise.
float shadowMapVisibility(float3 posW, int currLightIdx, int pointLightIndex)
{
    // Fixed samples for PCF of cube map
    // Based on https://learnopengl.com/Advanced-Lighting/Shadows/Point-Shadows
    static const int kCubePCFSamples = 20;
    static const float3 PCFOffsets[kCubePCFSamples] = {
        float3(1.f, 1.f, 1.f),  float3(1.f, -1.f, 1.f),  float3(-1.f, -1.f, 1.f),  float3(-1.f, 1.f, 1.f),
        float3(1.f, 1.f, -1.f), float3(1.f, -1.f, -1.f), float3(-1.f, -1.f, -1.f), float3(-1.f, 1.f, -1.f),
        float3(1.f, 1.f, 0.f),  float3(1.f, -1.f, 0.f),  float3(-1.f, -1.f, 0.f),  float3(-1.f, 1.f, 0.f),
        float3(1.f, 0.f, 1.f),  float3(-1.f, 0.f, 1.f),  float3(1.f, 0.f, -1.f),   float3(-1.f, 0.f, -1.f),
        float3(0.f, 1.f, 1.f),  float3(0.f, -1.f, 1.f),  float3(0.f, -1.f, -1.f),  float3(0.f, 1.f, -1.f)
    };
    // Fixed samples for PCF of directional shadow map
    // Based on http://www.opengl-tutorial.org/intermediate-tutorials/tutorial-16-shadow-mapping/
    static const int kDirPCFSamples = 5;
    static const float2 poissonDisk[kDirPCFSamples] = {
        float2(-0.94201624, -0.39906216),
        float2(0.94558609, -0.76890725),
        float2(-0.094184101, -0.92938870),
        float2(0.34495938, 0.29387760),
        float2(0., 0.)
    };
    static const float kGaussianKernel[3][3] = {
        { 0.0625, 0.125, 0.0625 },
        { 0.125, 0.25, 0.125 },
        { 0.0625, 0.125, 0.0625 }
    };

    // Not using shadow mapping - just return "visible"
    if (!gUsingShadowMapping) return 1.0f;

    // Directional shadow mapping 
    if (pointLightIndex == -1)
    {
        float4 posH = mul(float4(posW, 1.f), gDirViewProj); // Screen coords in range [-w, w]
        posH = posH / posH.w; // NDC in range [-1, 1]
        posH.y = -posH.y; // Falcor y-indices are flipped
        float actualDepth = posH.z / posH.w; // Get the actual deth
        float2 shadowMapCoords = posH.xy * 0.5f + 0.5f; // Transform to screen coords in range [0, 1]

        if (gUsingPCF)
        {
            float visibility = 0.0f;
            // Take the average of kDirPCFSamples visibiility samples from the shadow map at different offsets
            // as the visibility value. This allows us to have softer shadows.
            //for (int i = 0; i < kDirPCFSamples; i++)
            //{
            //    float shadowMapDepth = gDirShadowMap[(posH.xy + poissonDisk[i] * gDirPCFWidth) * gDirShadowMapRes];
            //    if (distNormalized <= shadowMapDepth + gDirShadowBias)
            //        visibility += 1.0;
            //}
            //return visibility / kDirPCFSamples;

            // Use a 3x3 Gaussian Kernel PCF
            for (int i = -1; i <= 1; i++)
            {
                for (int j = -1; j <= 1; j++)
                {
                    float shadowMapDepth = gDirShadowMap[(shadowMapCoords + float2(i, j) * gDirPCFWidth) * gDirShadowMapRes];
                    if (actualDepth <= shadowMapDepth + gDirShadowBias)
                        visibility += 1.0 * kGaussianKernel[i + 1][j + 1];
                }
            }
            return visibility;
        }
        else
        {
            float shadowMapDepth = gDirShadowMap[shadowMapCoords * gDirShadowMapRes];
            return actualDepth <= shadowMapDepth + gDirShadowBias;
        }
    }

    // Point light shadow mapping
    // Vector from the light to the fragment's location
    float3 viewVec = posW - gScene.getLight(currLightIdx).posW;
    // Get the actual depth value squared. We use square to prevent having to use sqrt.
    float actualDepth2 = dot(viewVec, viewVec);
    // Normalize this to [0, 1]
    float distNormalized = actualDepth2 / gCubeShadowFar2;

    // If it's beyond the far plane, just consider it not shadowed.
    if (distNormalized > 1.0) return 1.0f;

    if (gUsingPCF)
    {
        float visibility = 0.0f;
        // Take the average of kCubePCFSamples visibiility samples from the cubemap at different offsets
        // as the visibility value. This allows us to have softer shadows.
        for (int i = 0; i < kCubePCFSamples; i++)
        {
            // Get the cubemap coordinate to sample based on the normalized view vector direction (+ offset)
            float3 uvi = dirToCubeCoords(normalize(viewVec + gCubePCFWidth * PCFOffsets[i]));
            // Get the depth value in the shadow map. We add 6 * pointLightIndex
            // to the array index because each pointLight occupies 6 entries in the array of cubemaps
            float shadowMapDepth = gCubeShadowMap[uint3((uvi.xy + float2(0., uvi.z)) * uint2(gCubeShadowMapRes, gCubeShadowMapRes),
                                                        pointLightIndex)];
            if (distNormalized <= shadowMapDepth + gCubeShadowBias)
                visibility += 1.0;
        }
        return visibility / kCubePCFSamples;
    }
    else
    {
        // Get the cubemap coordinate to sample based on the normalized view vector direction
        float3 uvi = dirToCubeCoords(normalize(viewVec));
        // Get the depth value in the shadow map. We add 6 * pointLightIndex
        // to the array index because each pointLight occupies 6 entries in the array of cubemaps
        float shadowMapDepth = gCubeShadowMap[uint3((uvi.xy + float2(0., uvi.z)) * uint2(gCubeShadowMapRes, gCubeShadowMapRes),
                                                    pointLightIndex)];
        // Return 1.0 if the actual depth is what was seen by light
        return distNormalized <= shadowMapDepth + gCubeShadowBias;
    }
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
            if (distance(gLightLocations[lightIndex].xy * launchDim, launchIndex) < gLightRadius)
            {
                gOutput[launchIndex] = float4(frac(gLightLocations[lightIndex]), 1.0f);
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

        const uint lightCount = gScene.getLightCount();
        int pointLightIndex = -1;
        for (int lightIndex = 0; lightIndex < lightCount; lightIndex++)
        {
            bool isPointLight = gScene.getLight(lightIndex).type == uint32_t(LightType.Point);
            pointLightIndex += isPointLight ? 1 : 0;

            float distToLight;
            float3 lightIntensity;
            float3 toLight;
            // A helper (that queries the Falcor scene to get needed data about this light)
            getLightData(lightIndex, worldPos.xyz, toLight, lightIntensity, distToLight);

            // Compute our lambertion term (L dot N)
            float LdotN = saturate(dot(worldNorm.xyz, toLight));

            // Perform visibility check
            float shadowMult = shadowMapVisibility(worldPos.xyz, lightIndex,
                                                   isPointLight ? pointLightIndex : -1);

            // Compute our Lambertian shading color
            shadeColor += shadowMult * LdotN * lightIntensity;
        }

        // Physically based Lambertian term is albedo/pi
        shadeColor *= difMatlColor.rgb / M_PI;
    }
    // Just for debugging creation of shadowmap Will be removed for the
    // actual implementation, this is here for debugging purposes.
    else
    {
        float lum = gDirShadowMap[launchIndex];
        shadeColor = float3(lum, lum, lum);
    }

    // Save out our AO color. If we hit actual scene geometry, add the color contribution, otherwise
    // it's the environment map and we just write the color contribution directly.
    gOutput[launchIndex] = (worldPos.w != 0.0f)
        ? saturate(gOutput[launchIndex] + float4(shadeColor, 1.0f))
        : float4(shadeColor, 1.0f);
}
