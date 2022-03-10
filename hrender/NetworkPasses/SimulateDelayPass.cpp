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

#include "SimulateDelayPass.h"

char* SimulateDelayPass::getOutputBuffer()
{
	int index = (bufferPointer + mNumBuffers - mNumFramesDelay) % mNumBuffers;
	return buffers[index];
}

int SimulateDelayPass::getOutputBufferSize()
{
	int index = (bufferPointer + mNumBuffers - mNumFramesDelay) % mNumBuffers;
	return bufferSizes[index];
}

bool SimulateDelayPass::initialize(RenderContext* pRenderContext, ResourceManager::SharedPtr pResManager)
{
	for (int i = 0; i < mNumBuffers; i++) {
		char* buffer = new char[mMaxSize];
		buffers.push_back(buffer);
		bufferSizes.push_back(0);
	}
	
	return true;
}

void SimulateDelayPass::execute(RenderContext* pRenderContext)
{
	bufferPointer = (bufferPointer + 1) % mNumBuffers;
	int size = mGetInputBufferSize();
	memcpy(buffers[bufferPointer], mGetInputBuffer(), size);
	bufferSizes[bufferPointer] = size;
}

void SimulateDelayPass::renderGui(Gui::Window* pPassWindow)
{
	// Window is marked dirty if any of the configuration is changed.
	int dirty = 0;

	dirty |= (int)pPassWindow->var("Number of frames delayed", mNumFramesDelay, 0, mNumBuffers - 1, 0.01f);

	// If any of our UI parameters changed, let the pipeline know we're doing something different next frame
	if (dirty) setRefreshFlag();
}
