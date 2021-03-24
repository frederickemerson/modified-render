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

/** Ray traced ambient occlusion pass.
*/
class MemoryTransferPassClientCPU_GPU : public ::RenderPass
{
public:
    using SharedPtr = std::shared_ptr<MemoryTransferPassClientCPU_GPU>;
    using SharedConstPtr = std::shared_ptr<const MemoryTransferPassClientCPU_GPU>;

    static SharedPtr create() { return SharedPtr(new MemoryTransferPassClientCPU_GPU()); }
    virtual ~MemoryTransferPassClientCPU_GPU() = default;

protected:
    MemoryTransferPassClientCPU_GPU() : ::RenderPass("Memory Transfer Pass", "Memory Transfer Pass Options") { }

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
};
