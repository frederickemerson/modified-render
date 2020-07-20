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

import Scene.ShadingData;

/** Convert float value to 8-bit unorm (unsafe version).
    \param[in] v Value assumed to be in [0,1].
    \return 16-bit unorm in low bits, high bits all zeros.
*/
uint packUnorm8_unsafe(float v)
{
    // + 0.5f is to ensure that truncation doesn't drop to the next
    // integer due to precision issues, e.g. 4.999 -> 4
    return (uint)trunc(v * 255.f + 0.5f);
}

/** Convert float value to 8-bit unorm.
    Values outside [0,1] are clamped and NaN is encoded as zero.
    \return 16-bit unorm in low bits, high bits all zeros.
*/
uint packUnorm8(float v)
{
    v = isnan(v) ? 0.f : saturate(v);
    return packUnorm8_unsafe(v);
}

/** Pack four floats into 8-bit unorm values in a dword.
*/
uint packUnorm4x8(float4 v)
{
    return (packUnorm8(v.w) << 24) | (packUnorm8(v.z) << 16) |
        (packUnorm8(v.y) << 8) | packUnorm8(v.x);
}

/** Pack two floats into 16-bit unorm values in a dword (unsafe version)
    \param[in] v Two values assumed to be in [0,1].
*/
uint packUnorm4x8_unsafe(float4 v)
{
    return (packUnorm8_unsafe(v.w) << 24) | (packUnorm8_unsafe(v.z) << 16) |
        (packUnorm8_unsafe(v.y) << 8) | packUnorm8_unsafe(v.x);
}

/** Convert 16-bit unorm to float value.
    \param[in] packed 16-bit unorm in low bits, high bits don't care.
    \return Float value in [0,1].
*/
float unpackUnorm8(uint packed)
{
    return float(packed & 0xff) * (1.f / 255);
}

/** Unpack two 16-bit unorm values from a dword.
*/
float4 unpackUnorm4x8(uint packed)
{
    return float4(packed & 0xff, (packed >> 8) & 0xff,
        (packed >> 16) & 0xff, (packed >> 24) & 0xff) * (1.f / 255);
}

/** Extract the texture data in the ShadingData provided into a float4.
    \param[in] hitPt ShadingData of the hit point as specified in ShadingData.slang
    \return A single uint4, with a packed format with 8 bits per component:
        r: diffuse.r,    diffuse.g,      diffuse.b,          opacity
        g: specular.r,   specular.g,     specular.b,         linear roughness
        b: emissive.r,   emissive.g,     emissive.b,         doubleSided ? 1.0f : 0.0f
        a: IoR,          metallic,       specular trans,     eta
*/
uint4 packTextureData(in ShadingData hitPt)
{
    return uint4(
        packUnorm4x8(float4(hitPt.diffuse, hitPt.opacity)),
        packUnorm4x8(float4(hitPt.specular, hitPt.linearRoughness)),
        packUnorm4x8(float4(hitPt.emissive, hitPt.doubleSided ? 1.f : 0.f)),
        packUnorm4x8(float4(hitPt.IoR, hitPt.metallic, hitPt.specularTransmission, hitPt.eta))
    );
}

/** Extract and unpack the texture data in packedData to populate the material vectors.
    \param[in] packedData Texture data packed by packTextureData
*/
void unpackTextureData(in uint4 packedData, out float4 matDif, out float4 matSpec,
    out float4 matEmissiveDoubleSided, out float4 matExtra)
{
    matDif                 = unpackUnorm4x8(packedData.x);
    matSpec                = unpackUnorm4x8(packedData.y);
    matEmissiveDoubleSided = unpackUnorm4x8(packedData.z);
    matExtra               = unpackUnorm4x8(packedData.w);
}
