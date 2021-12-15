#pragma once

#include "lz4.h"
#include "../DxrTutorSharedUtils/RenderPass.h"

/**
 * Transfer data from server to client or client to server
 * based on the configuration setting.
 */
class CompressionPass : public ::RenderPass
{

public:
    enum class Mode
    {
        Compression = 0,
        Decompression = 1
    };
    using SharedPtr = std::shared_ptr<CompressionPass>;
    using SharedConstPtr = std::shared_ptr<const CompressionPass>;
    virtual ~CompressionPass() = default;

    // Buffer for storing output of compression/decompression
    char* outputBuffer;

    static SharedPtr create(Mode mode) {
        if (mode == Mode::Compression) {
            return SharedPtr(new CompressionPass(mode, "Compression Pass", "Compression Pass Gui"));
        }
        else {
            return SharedPtr(new CompressionPass(mode, "Decompression Pass", "Decompression Pass Gui"));
        }
    }

protected:
    CompressionPass(Mode mode, const std::string name = "<Unknown render pass>", const std::string guiName = "<Unknown gui group>") :RenderPass(name, guiName) {
        mMode = mode;
    }

    // Implementation of RenderPass interface
    bool initialize(RenderContext* pRenderContext, ResourceManager::SharedPtr pResManager) override;
    void initScene(RenderContext* pRenderContext, Scene::SharedPtr pScene) override;
    void execute(RenderContext* pRenderContext) override;
    void renderGui(Gui::Window* pPassWindow) override;

    Mode                                    mMode;                     ///< Whether this pass runs as compression or decompression
};