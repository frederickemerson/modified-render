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
#include "../DxrTutorSharedUtils/Regression.h"

/**
 * Memory transfer on client side, from CPU to GPU.
 */
class MemoryTransferPassClientCPU_GPU : public ::RenderPass
{
public:
    using SharedPtr = std::shared_ptr<MemoryTransferPassClientCPU_GPU>;
    using SharedConstPtr = std::shared_ptr<const MemoryTransferPassClientCPU_GPU>;

    static SharedPtr create(std::function<char* ()> getInputBuffer) { 
        return SharedPtr(new MemoryTransferPassClientCPU_GPU(getInputBuffer)); 
    }
    virtual ~MemoryTransferPassClientCPU_GPU() = default;

protected:
    MemoryTransferPassClientCPU_GPU(std::function<char* ()> getInputBuffer) : ::RenderPass("Memory Transfer Pass Client CPU-GPU", "Memory Transfer Pass Options") {
        mGetInputBuffer = getInputBuffer;
    }

    // Implementation of RenderPass interface
    bool initialize(RenderContext* pRenderContext, ResourceManager::SharedPtr pResManager) override;
    void initScene(RenderContext* pRenderContext, Scene::SharedPtr pScene) override;
    void execute(RenderContext* pRenderContext) override;
    void renderGui(Gui::Window* pPassWindow) override;

    // Get the texture data from the GPU into a RAM array
    std::vector<uint8_t> texData(RenderContext* pRenderContext, Texture::SharedPtr tex);

    // Override some functions that provide information to the RenderPipeline class
    bool requiresScene() override { return true; }

    // Rendering state
    Scene::SharedPtr                        mpScene;                ///< Our scene file (passed in from app)

    // index of textures we will be accessing
    int32_t mVisibilityIndex = -1;                                  ///< index of visibility texture, to be obtained in initialization

    // Function for getting input buffers
    std::function<char* ()> mGetInputBuffer;
};
