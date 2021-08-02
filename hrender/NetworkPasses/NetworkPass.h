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
class NetworkPass : public ::RenderPass
{

public:
    enum class Mode
    {
        Client = 0,
        ClientSend = 1,
        Server = 2,
        ServerSend = 3,
    };
    using SharedPtr = std::shared_ptr<NetworkPass>;
    using SharedConstPtr = std::shared_ptr<const NetworkPass>;

    static SharedPtr create(Mode mode = Mode::Client, int texWidth = -1, int texHeight = -1) {
        return SharedPtr(new NetworkPass(mode, texWidth, texHeight));
    }
    virtual ~NetworkPass() = default;

    // Texture data from transfering
    static std::vector<uint8_t> posData;
    static int posTexWidth;
    static int posTexHeight;

    static std::vector<uint8_t> visibilityData;
    static std::array<float3, 3> camData;

protected:
    NetworkPass(Mode mode, int texWidth = -1, int texHeight = -1) : ::RenderPass("Network Pass", "Network Pass Options") {
        mMode = mode; mTexWidth = texWidth; mTexHeight = texHeight;
    }

    // Implementation of RenderPass interface
    bool initialize(RenderContext* pRenderContext, ResourceManager::SharedPtr pResManager) override;
    void initScene(RenderContext* pRenderContext, Scene::SharedPtr pScene) override;
    void execute(RenderContext* pRenderContext) override;
    void renderGui(Gui::Window* pPassWindow) override;
    // Different execution functions
    void executeClientSend(RenderContext* pRenderContext);
    void executeClientRecv(RenderContext* pRenderContext);
    void executeServerSend(RenderContext* pRenderContext);
    void executeServerRecv(RenderContext* pRenderContext);
    bool firstClientRender(RenderContext* pRenderContext);
    // For first client render on UDP, send the client's window width and height
    bool firstClientRenderUdp(RenderContext* pRenderContext);
    bool firstServerRender(RenderContext* pRenderContext);

    // Get the texture data from the GPU into a RAM array
    std::vector<uint8_t> texData(RenderContext* pRenderContext, Texture::SharedPtr tex);

    // Override some functions that provide information to the RenderPipeline class
    bool requiresScene() override { return true; }
    bool usesRayTracing() override { return true; }

    // Rendering state
    RayLaunch::SharedPtr                    mpRays;                 ///< Our wrapper around a DX Raytracing pass
    Scene::SharedPtr                        mpScene;                ///< Our scene file (passed in from app)
    // Various internal parameters
    Mode                                    mMode;                  ///< Whether this pass runs as client or server
    bool                                    mFirstRender = true;    ///< If this is the first time rendering, need to send scene
    int32_t                                 mOutputIndex;           ///< An index for our output buffer
    std::string                             mOutputTexName;         ///< Where do we want to store the results?
    int                                     mTexWidth = -1;         ///< The width of the texture we render, based on the client
    int                                     mTexHeight = -1;        ///< The height of the texture we render, based on the client

    bool firstClientReceive = true; // Use a longer timeout for first client receive
};

