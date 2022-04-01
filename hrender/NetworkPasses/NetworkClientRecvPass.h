#pragma once

#include "NetworkPass.h"

class NetworkClientRecvPass : public :: NetworkPass {
public:
	using SharedPtr = std::shared_ptr<NetworkClientRecvPass>;

	static SharedPtr create(int texWidth, int texHeight) {
		return SharedPtr(new NetworkClientRecvPass(texWidth, texHeight));
	}
	const NetworkPass::Mode mode = Mode::ClientRecv;

	// Client - Two buffers for writing and reading at the same time
	static char* clientReadBuffer;
	static char* clientWriteBuffer;
	static char* intermediateBuffer; // for storing before compression/decompression

protected:
	NetworkClientRecvPass(int texWidth, int texHeight) 
		: NetworkPass(texWidth, texHeight, "Network Client Recv Pass", "Network Client Recv Pass Options") {
	}

	void execute(RenderContext* pRenderContext) override;
	void renderGui(Gui::Window* pPassWindow) override;
	void checkSequential(bool bSwitching);

	bool bSwitching = true;
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

	// camera weights
	float cameraWeightUX = 1;
	float cameraWeightUY = 1;
	float cameraWeightUZ = 1;
	float cameraWeightVX = 1;
	float cameraWeightVY = 1;
	float cameraWeightVZ = 1;
	float cameraWeightWX = 1;
	float cameraWeightWY = 1;
	float cameraWeightWZ = 1;
	float networkWeight = 1;

	// threshold values
	int lowThreshold = 80;
	int midThreshold = 400;
	int highThreshold = 1000;
	const int timeToRemainInSequential = 20; // frames

	float totalCameraChange;
};