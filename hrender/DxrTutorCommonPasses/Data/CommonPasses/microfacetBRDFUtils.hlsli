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
#include "Utils/Math/MathConstants.slangh"

float3 getReflectionVec(float3 H, float3 V)
{
	return normalize(2.f * dot(V, H) * H - V);
}

// The NDF for GGX, see Eqn 19 from 
//    http://blog.selfshadow.com/publications/s2012-shading-course/hoffman/s2012_pbs_physics_math_notes.pdf
//
// This function can be used for "D" in the Cook-Torrance model:  D*G*F / (4*NdotL*NdotV)
float ggxNormalDistribution(float NdotH, float roughness)
{
	float a2 = roughness * roughness;
	float d = ((NdotH * a2 - NdotH) * NdotH + 1);
	return a2 / max(0.001f, (d * d * M_PI));
}

// This from Schlick 1994, modified as per Karas in SIGGRAPH 2013 "Physically Based Shading" course
//
// This function can be used for "G" in the Cook-Torrance model:  D*G*F / (4*NdotL*NdotV)
float ggxSchlickMaskingTerm(float NdotL, float NdotV, float roughness)
{
	// Karis notes they use alpha / 2 (or roughness^2 / 2)
	float k = roughness*roughness / 2;

	// Karis also notes they can use the following equation, but only for analytical lights
	//float k = (roughness + 1)*(roughness + 1) / 8; 

	// Compute G(v) and G(l).  These equations directly from Schlick 1994
	//     (Though note, Schlick's notation is cryptic and confusing.)
	float g_v = NdotV / (NdotV*(1 - k) + k);
	float g_l = NdotL / (NdotL*(1 - k) + k);

	// Return G(v) * G(l)
	return g_v * g_l;
}

// Traditional Schlick approximation to the Fresnel term (also from Schlick 1994)
//
// This function can be used for "F" in the Cook-Torrance model:  D*G*F / (4*NdotL*NdotV)
float3 schlickFresnel(float3 f0, float u)
{
	return f0 + (float3(1.0f, 1.0f, 1.0f) - f0) * pow(1.0f - u, 5.0f);
}

// Get a GGX half vector / microfacet normal, sampled according to the distribution computed by
//     the function ggxNormalDistribution() above.  
//
// When using this function to sample, the probability density is pdf = D * NdotH / (4 * HdotV)
float3 getGGXMicrofacet(inout uint randSeed, float roughness, float3 hitNorm)
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
	return T * (sinThetaH * cos(phiH)) + B * (sinThetaH * sin(phiH)) + hitNorm * cosThetaH;
}

// Compute GGX term from input parameters
float3 getGGXColor(float3 V, float3 L, float3 N, float3 specColor, float roughness)
{
	// Compute half vector and dot products
	float3 H = normalize(V + L);
	float NdotL = saturate(dot(N, L)); // Lambertian term
	float NdotH = saturate(dot(N, H));
	float LdotH = saturate(dot(L, H));
	float NdotV = saturate(dot(N, V));

	// Evaluate our GGX BRDF term
	float  D = ggxNormalDistribution(NdotH, roughness);
	float  G = ggxSchlickMaskingTerm(NdotL, NdotV, roughness);
	float3 F = schlickFresnel(specColor, LdotH);

	// Evaluate the Cook-Torrance Microfacet BRDF model
	float3 outColor = D * G * F / (4.f * NdotV /* * NdotL */);

	// Determine if the color is valid (if invalid, we likely have a NaN or Inf)
	bool invalid = NdotV * NdotL * LdotH <= 0.0f;
	return invalid ? float3(0, 0, 0) : outColor;
}

// Compute GGX term from input parameters
void getGGXColorAndProb(inout uint randSeed, float3 V, float3 N, float3 specColor, float roughness,
	out float3 ggxBRDF, out float ggxProb, out float3 L, out float NdotL)
{
	// Randomly sample the NDF to get a microfacet in our BRDF to reflect off of
	float3 H = getGGXMicrofacet(randSeed, roughness, N);
	
	// Compute the outgoing direction based on this (perfectly reflective) microfacet
	L = getReflectionVec(H, V);
	
	// Compute dot products
	NdotL = saturate(dot(N, L)); // Lambertian term
	float NdotH = saturate(dot(N, H));
	float LdotH = saturate(dot(L, H));
	float NdotV = saturate(dot(N, V));

	// Evaluate our GGX BRDF term
	float  D = ggxNormalDistribution(NdotH, roughness);
	float  G = ggxSchlickMaskingTerm(NdotL, NdotV, roughness);
	float3 F = schlickFresnel(specColor, LdotH);

	// Compute the results. Common terms are cancelled.
    //ggxBRDF = D * G * F / ( 4.f * NdotV  * NdotL ); 
    //ggxProb = D * NdotH / ( 4.f * LdotH); 
	ggxBRDF = G * F / (NdotV); // The Cook-Torrance microfacet BRDF
	ggxProb = NdotH / (LdotH);         // The probability of sampling vector H from getGGXMicrofacet()

	// Determine if the color is valid (if invalid, we likely have a NaN or Inf)
	//bool invalid = NdotV * NdotL * LdotH <= 0.0f;
	//ggxBRDF = invalid ? float3(0, 0, 0) : ggxBRDF;
	//ggxProb = invalid ? 0.f : ggxProb;
}
