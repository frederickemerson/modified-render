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

/**
 * Transfer data from server to client or client to server
 * based on the configuration setting.
 */
class NetworkPass : public :: RenderPass
{

public:
    enum class Mode
    {
        ClientRecv = 0,
        ClientSend = 1,
        ServerRecv = 2,
        ServerSend = 3
    };
    using SharedPtr = std::shared_ptr<NetworkPass>;
    using SharedConstPtr = std::shared_ptr<const NetworkPass>;
    virtual ~NetworkPass() = default;

    // Texture data from transfering
    static std::vector<uint8_t> posData;
    static int posTexWidth;
    static int posTexHeight;

    // Client - Two buffers for writing and reading at the same time
    static char* visibilityDataForReadingClient;
    static char* visibilityDataForWritingClient;
    // Server - Just one pointer to a Falcor Buffer that allows for reading safely
    // This Buffer will be set by MemoryTransferPassClientGPU_CPU
    // for server side GPU-CPU trsf of visibilityBuffer, stores location of data, changes every frame
    static uint8_t* pVisibilityDataServer;
    static std::array<float3, 3> camData;

protected:
    NetworkPass(const std::string name = "<Unknown render pass>", const std::string guiName = "<Unknown gui group>") :RenderPass(name, guiName) {}

    // Implementation of RenderPass interface
    bool initialize(RenderContext* pRenderContext, ResourceManager::SharedPtr pResManager) override;
    void initScene(RenderContext* pRenderContext, Scene::SharedPtr pScene) override;
    void renderGui(Gui::Window* pPassWindow) override;

    // Override some functions that provide information to the RenderPipeline class
    bool requiresScene() override { return true; }
    bool usesRayTracing() override { return true; }

    // Rendering state
    RayLaunch::SharedPtr                    mpRays;                    ///< Our wrapper around a DX Raytracing pass
    Scene::SharedPtr                        mpScene;                   ///< Our scene file (passed in from app)
    // Various internal parameters
    Mode                                    mMode;                     ///< Whether this pass runs as client or server
    int32_t                                 mOutputIndex;              ///< An index for our output buffer
    std::string                             mOutputTexName;            ///< Where do we want to store the results?


};

