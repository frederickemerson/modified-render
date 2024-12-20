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
#include "../DxrTutorSharedUtils/RenderPass.h"
#include "../DxrTutorSharedUtils/RayLaunch.h"
#include "../NetworkPasses/NetworkPass.h"

#include "../DxrTutorSharedUtils/RenderConfig.h"

/**
 * Memory transfer on server side, from GPU to CPU.
 */
class MemoryTransferPassServerGPU_CPU : public ::RenderPass
{
public:
    using SharedPtr = std::shared_ptr<MemoryTransferPassServerGPU_CPU>;
    using SharedConstPtr = std::shared_ptr<const MemoryTransferPassServerGPU_CPU>;

    static SharedPtr create(bool isHybridRendering) { return SharedPtr(new MemoryTransferPassServerGPU_CPU(isHybridRendering)); }
    virtual ~MemoryTransferPassServerGPU_CPU() = default;

    // get output buffer on CPU memory after memory transfer
    char* getOutputBuffer() { return (char*)outputBuffer; }
    int getOutputBufferSize() { return mHybridMode ? VIS_TEX_LEN + AO_TEX_LEN + REF_TEX_LEN : VIS_TEX_LEN; } // Remote uses same size as VIS_TEX_LEN

protected:
    MemoryTransferPassServerGPU_CPU(bool isHybridRendering) : ::RenderPass("Memory Transfer Pass Server GPU-CPU", "Memory Transfer Pass Options") {
        mHybridMode = isHybridRendering;
    }

    // Implementation of RenderPass interface
    bool initialize(RenderContext* pRenderContext, ResourceManager::SharedPtr pResManager) override;
    void initScene(RenderContext* pRenderContext, Scene::SharedPtr pScene) override;
    void execute(RenderContext* pRenderContext) override;
    void renderGui(Gui::Window* pPassWindow) override;

    // Override some functions that provide information to the RenderPipeline class
    bool requiresScene() override { return true; }

    // Rendering state
    Scene::SharedPtr mpScene;                ///< Our scene file (passed in from app)

    // index of textures we will be accessing
    int32_t mVisibilityIndex = -1;           ///< index of visibility texture, to be obtained in initialization
    int32_t mSRTReflectionsIndex = -1;       ///< index of reflections texture, to be obtained in initialization
    int32_t mAOIndex = -1;                   ///< index of ambient occlusion texture, to be obtained in initialization
    int32_t mVShadingIndex = -1;             ///< index of v-shading, to be obtained in initialization only for remote
    int32_t mGIIndex = -1;                   ///< index of global illumination texture, current unused

    bool mHybridMode = true;                 ///< True if doing hybrid rendering, else remote rendering.

    // initialise output buffer
    uint8_t* outputBuffer;
    
};
