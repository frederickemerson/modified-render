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
#include <vector>
#include "../DxrTutorSharedUtils/RenderPass.h"

class SimulateDelayPass : public ::RenderPass {
public:
	using SharedPtr = std::shared_ptr<SimulateDelayPass>;

	static SharedPtr create(std::function<char* ()> getInputBuffer, std::function<int()> getInputBufferSize,
		int maxSize = 4*1920*1080, int numBuffers = 4, int numFramesDelay = 1) {
		return SharedPtr(new SimulateDelayPass(getInputBuffer, getInputBufferSize, maxSize, numBuffers, numFramesDelay));
	}

	char* getOutputBuffer();
	int getOutputBufferSize();

protected:
	SimulateDelayPass(std::function<char* ()> getInputBuffer, std::function<int()> getInputBufferSize,
		int maxSize, int numBuffers, int numFramesDelay)
		: ::RenderPass("Simulate Delay Pass", "Simulate Delay Pass Options") {
		mGetInputBuffer = getInputBuffer;
		mGetInputBufferSize = getInputBufferSize;
		mMaxSize = maxSize;
		mNumBuffers = numBuffers;
		mNumFramesDelay = numFramesDelay;
	}

	bool initialize(RenderContext* pRenderContext, ResourceManager::SharedPtr pResManager) override;
	void execute(RenderContext* pRenderContext) override;
	void renderGui(Gui::Window* pPassWindow) override;


protected:
	std::function<char* ()> mGetInputBuffer;
	std::function<int()> mGetInputBufferSize;
	int mNumFramesDelay;
	int mMaxSize;
	int mNumBuffers;
	
	int bufferPointer = 0;
	std::vector<char*> buffers;
	std::vector<int> bufferSizes;
};