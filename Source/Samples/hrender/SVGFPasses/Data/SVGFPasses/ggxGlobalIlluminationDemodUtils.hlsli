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

// Define pi
#define M_1_PI  0.318309886183790671538

// A helper to extract important light data from internal Falcor data structures.  What's going on isn't particularly
//     important -- any framework you use will expose internal scene data in some way.  Use your framework's utilities.
void getLightData(in int index, in float3 hitPos, out float3 toLight, out float3 lightIntensity, out float distToLight)
{
    // Use built-in Falcor functions and data structures to fill in a LightSample data structure
    //   -> See "Lights.slang" for it's definition
    LightSample ls;

    // Is it a directional light?
    if (gLights[index].type == LightDirectional)
        ls = evalDirectionalLight(gLights[index], hitPos);

    // No?  Must be a point light.
    else
        ls = evalPointLight(gLights[index], hitPos);

    // Convert the LightSample structure into simpler data
    toLight = normalize(ls.L);
    lightIntensity = ls.diffuse;
    distToLight = length(ls.posW - hitPos);
}

// Encapsulates a bunch of Falcor stuff into one simpler function. 
//    -> This can only be called within a closest hit or any hit shader
ShadingData getHitShadingData( BuiltinIntersectionAttribs attribs )
{
    // Run a pair of Falcor helper functions to compute important data at the current hit point
    VertexOut  vsOut = getVertexAttributes(PrimitiveIndex(), attribs);
    return prepareShadingData(vsOut, gMaterial, gCamera.posW);
}

// Utility function to get a vector perpendicular to an input vector 
//    (from "Efficient Construction of Perpendicular Vectors Without Branching")
float3 getPerpendicularVector(float3 u)
{
    float3 a = abs(u);
    uint xm = ((a.x - a.y)<0 && (a.x - a.z)<0) ? 1 : 0;
    uint ym = (a.y - a.z)<0 ? (1 ^ xm) : 0;
    uint zm = 1 ^ (xm | ym);
    return cross(u, float3(xm, ym, zm));
}

// A work-around function because some DXR drivers seem to have broken atan2() implementations
float atan2_WAR(float y, float x)
{
    if (x > 0.f)
        return atan(y / x);
    else if (x < 0.f && y >= 0.f)
        return atan(y / x) + M_PI;
    else if (x < 0.f && y < 0.f)
        return atan(y / x) - M_PI;
    else if (x == 0.f && y > 0.f)
        return M_PI / 2.f;
    else if (x == 0.f && y < 0.f)
        return -M_PI / 2.f;
    return 0.f; // x==0 && y==0 (undefined)
}

// Convert our world space direction to a (u,v) coord in a latitude-longitude spherical map
float2 wsVectorToLatLong(float3 dir)
{
    float3 p = normalize(dir);

    // atan2_WAR is a work-around due to an apparent compiler bug in atan2
    float u = (1.f + atan2_WAR(p.x, -p.z) * M_1_PI) * 0.5f;
    float v = acos(p.y) * M_1_PI;
    return float2(u, v);
}

// Generates a seed for a random number generator from 2 inputs plus a backoff
uint initRand(uint val0, uint val1, uint backoff = 16)
{
    uint v0 = val0, v1 = val1, s0 = 0;

    [unroll]
    for (uint n = 0; n < backoff; n++)
    {
        s0 += 0x9e3779b9;
        v0 += ((v1 << 4) + 0xa341316c) ^ (v1 + s0) ^ ((v1 >> 5) + 0xc8013ea4);
        v1 += ((v0 << 4) + 0xad90777d) ^ (v0 + s0) ^ ((v0 >> 5) + 0x7e95761e);
    }
    return v0;
}

// Takes our seed, updates it, and returns a pseudorandom float in [0..1]
float nextRand(inout uint s)
{
    s = (1664525u * s + 1013904223u);
    return float(s & 0x00FFFFFF) / float(0x01000000);
}

// Get a cosine-weighted random vector centered around a specified normal direction.
float3 getCosHemisphereSample(inout uint randSeed, float3 hitNorm)
{
    // Get 2 random numbers to select our sample with
    float2 randVal = float2(nextRand(randSeed), nextRand(randSeed));

    // Cosine weighted hemisphere sample from RNG
    float3 bitangent = getPerpendicularVector(hitNorm);
    float3 tangent = cross(bitangent, hitNorm);
    float r = sqrt(randVal.x);
    float phi = 2.0f * 3.14159265f * randVal.y;

    // Get our cosine-weighted hemisphere lobe sample direction
    return tangent * (r * cos(phi).x) + bitangent * (r * sin(phi)) + hitNorm.xyz * sqrt(max(0.0, 1.0f - randVal.x));
}

// This function tests if the alpha test fails, given the attributes of the current hit. 
//   -> Can legally be called in a DXR any-hit shader or a DXR closest-hit shader, and 
//      accesses Falcor helpers and data structures to extract and perform the alpha test.
bool alphaTestFails(BuiltinIntersectionAttribs attribs)
{
    // Run a Falcor helper to extract the current hit point's geometric data
    VertexOut  vsOut = getVertexAttributes(PrimitiveIndex(), attribs);

    // Extracts the diffuse color from the material (the alpha component is opacity)
    float4 baseColor = sampleTexture(gMaterial.resources.baseColor, gMaterial.resources.samplerState,
        vsOut.texC, gMaterial.baseColor, EXTRACT_DIFFUSE_TYPE(gMaterial.flags));

    // Test if this hit point fails a standard alpha test.  
    return (baseColor.a < gMaterial.alphaThreshold);
}

// The NDF for GGX, see Eqn 19 from 
//    http://blog.selfshadow.com/publications/s2012-shading-course/hoffman/s2012_pbs_physics_math_notes.pdf
float ggxNormalDistribution(float NdotH, float roughness)
{
    float a2 = roughness * roughness;
    float d = ((NdotH * a2 - NdotH) * NdotH + 1);
    return a2 / max(0.001f, (d * d * M_PI));
}

// The correlated version from 
//     http://jcgt.org/published/0003/02/03/paper.pdf 
float ggxSmithMaskingTerm(float NdotL, float NdotV, float roughness)
{
    float a2 = roughness * roughness;
    float lambdaV = NdotL * sqrt(max(0.0f, (-NdotV * a2 + NdotV) * NdotV + a2));
    float lambdaL = NdotV * sqrt(max(0.0f, (-NdotL * a2 + NdotL) * NdotL + a2));
    return 0.5f / (lambdaV + lambdaL);
}

// Traditional Schlick approximation to the Fresnel term
float3 schlickFresnel(float3 f0, float u)
{
    return f0 + (float3(1.0f, 1.0f, 1.0f) - f0) * pow(1.0f - u, 5.0f);
}

// Compute GGX term from input parameters
float3 getGGXColor(float3 V, float3 L, float3 N, float NdotV, float3 specColor, float roughness, bool evalDirect)
{
    // Compute half vector and dot products
    float3 H = normalize(V + L);
    float NdotL = saturate(dot(N, L));
    float NdotH = saturate(dot(N, H));
    float LdotH = saturate(dot(L, H));

    // Evaluate our GGX BRDF term
    float D = ggxNormalDistribution(NdotH, roughness);
    float G = ggxSmithMaskingTerm(NdotL, NdotV, roughness) * 4 * NdotL;
    float3 F = schlickFresnel(specColor, LdotH);

    // If this is direct illumination, color is simple. 
    // If we sampled via getGGXSample(), we need to divide by the probability of this sample.
    float3 outColor = evalDirect ?
        F * G * D * NdotV :
        F * G * LdotH / max(0.001f, NdotH);

    // Determine if the color is valid (if invalid, we likely have a NaN or Inf)
    return (NdotV * NdotL * LdotH <= 0.0f) ? float3(0, 0, 0) : outColor;
}

float3 getGGXSampleDir(inout uint randSeed, float roughness, float3 hitNorm, float3 inVec)
{
    // Get our uniform random numbers
    float2 randVal = float2(nextRand(randSeed), nextRand(randSeed));

    // Get an orthonormal basis from the normal
    float3 B = getPerpendicularVector(hitNorm);
    float3 T = cross(B, hitNorm);

    // GGX NDF sampling
    float a2 = roughness * roughness;
    float cosThetaH = sqrt(max(0.0f, (1.0 - randVal.x) / ((a2 - 1.0) * randVal.x + 1)));
    float sinThetaH = sqrt(max(0.0f, 1.0f - cosThetaH * cosThetaH));
    float phiH = randVal.y * M_PI * 2.0f;

    // Get our GGX NDF sample (i.e., the half vector)
    float3 H = T * (sinThetaH * cos(phiH)) + B * (sinThetaH * sin(phiH)) + hitNorm * cosThetaH;

    // Convert this into a ray direction by computing the reflection direction
    return normalize(2.f * dot(inVec, H) * H - inVec);
}

float probabilityToSampleDiffuse(float3 difColor, float3 specColor)
{
    float lumDiffuse = max(0.01f, luminance(difColor.rgb));
    float lumSpecular = max(0.01f, luminance(specColor.rgb));
    return lumDiffuse / (lumDiffuse + lumSpecular);
}
