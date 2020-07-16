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

// The correlated version from 
//     http://jcgt.org/published/0003/02/03/paper.pdf 
float ggxSmithMaskingTerm(float NdotL, float NdotV, float roughness)
{
    float a2 = roughness * roughness;
    float lambdaV = NdotL * sqrt(max(0.0f, (-NdotV * a2 + NdotV) * NdotV + a2));
    float lambdaL = NdotV * sqrt(max(0.0f, (-NdotL * a2 + NdotL) * NdotL + a2));
    return 0.5f / (lambdaV + lambdaL);
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
    // float G = ggxSmithMaskingTerm(NdotL, NdotV, roughness) * 4 * NdotL; // Original
    // float G = ggxSmithMaskingTerm(NdotL, NdotV, roughness) / (4 * NdotL); // Original with modification
    float G = ggxSchlickMaskingTerm(NdotL, NdotV, roughness) / (4 * NdotL); // Using the new version's G estimate
    float3 F = schlickFresnel(specColor, LdotH);

    // If this is direct illumination, color is simple. 
    // If we sampled via getGGXSample(), we need to divide by the probability of this sample.
    float3 outColor = evalDirect ?
        // F * G * D * NdotV : // Original
        F * G * D / NdotV :
        // F * G * LdotH / max(0.001f, NdotH); // Original
        F * G * LdotH * 4.f / max(0.001f, NdotH) / NdotV;

    // Determine if the color is valid (if invalid, we likely have a NaN or Inf)
    return (NdotV * NdotL * LdotH <= 0.0f) ? float3(0, 0, 0) : outColor;
}
