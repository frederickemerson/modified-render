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


Texture2D<float4> gSSRColor;
Texture2D<float4> gSRTColor;
Texture2D<float4> gVshading;

cbuffer RCCB
{
	bool     gSkipRC;
	float3   gCamPos;
}

struct PS_OUTPUT
{
	float4 RCColor  : SV_Target0;
};

PS_OUTPUT main(float2 texC : TEXCOORD, float4 pos : SV_Position)
{
	PS_OUTPUT RCBufOut;
	uint2 pixelPos = (uint2)pos.xy;

	// Load VColor, SSRColor and SRTColor
	float4 VColor = gVshading[pixelPos];
	float4 SSRColor = gSSRColor[pixelPos];
	float4 SRTColor = gSRTColor[pixelPos];
	float3 shadeColor = VColor.rgb + SSRColor.rgb + 0.5f*SRTColor.rgb;


	if (shadeColor.r >= 1.0f)shadeColor.r = 1.0f;
	else if (shadeColor.g >= 1.0f)shadeColor.g = 1.0f;
	else if (shadeColor.b >= 1.0f)shadeColor.b = 1.0f;


	RCBufOut.RCColor = float4(shadeColor, 1.0f);

	return RCBufOut;
}