#pragma once

#include "NetworkPass.h"

class NetworkClientRecvPass : public :: NetworkPass {
public:
	using SharedPtr = std::shared_ptr<NetworkClientRecvPass>;

	static SharedPtr create(int texWidth, int texHeight) {
		return SharedPtr(new NetworkClientRecvPass(texWidth, texHeight));
	}
	const NetworkPass::Mode mode = Mode::ClientRecv;

protected:
	NetworkClientRecvPass(int texWidth, int texHeight) 
		: NetworkPass(texWidth, texHeight, "Network Client Recv Pass", "Network Client Recv Pass Options") {
	}

	void execute(RenderContext* pRenderContext) override;
	void renderGui(Gui::Window* pPassWindow) override;
	void checkMotionVector();
	void checkNetworkPing();

	bool sequential = false;
	int remainInSequential = 0;
	bool                                    firstClientReceive = true; // Use a longer timeout for first client receive

	// camera data
	float cameraUX = 0;
	float cameraUY = 0;
	float cameraUZ = 0;
	float cameraVX = 0;
	float cameraVY = 0;
	float cameraVZ = 0;
	float cameraWX = 0;
	float cameraWY = 0;
	float cameraWZ = 0;
};