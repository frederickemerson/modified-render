#pragma once

#include "NetworkPass.h"

class NetworkClientSendPass : public :: NetworkPass {
public:
	using SharedPtr = std::shared_ptr<NetworkClientSendPass>;

	static SharedPtr create(int texWidth, int texHeight) {
		return SharedPtr(new NetworkClientSendPass(texWidth, texHeight));
	}
	const NetworkPass::Mode mode = Mode::ClientSend;

protected:
	NetworkClientSendPass(int texWidth, int texHeight) 
		: NetworkPass(texWidth, texHeight, "Network Client Send Pass", "Network Client Send Pass Options") {}

	void execute(RenderContext* pRenderContext) override;
    void renderGui(Gui::Window* pPassWindow) override;
	// For first client render on UDP, send the client's window width and height
	bool firstClientRenderUdp(RenderContext* pRenderContext);

protected:
	bool                                    mFirstRender = true;       ///< If this is the first time rendering, need to send scene
	int                                     mArtificialDelay = 0;
};