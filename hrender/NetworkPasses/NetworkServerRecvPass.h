#pragma once

#include "NetworkPass.h"

class NetworkServerRecvPass : public :: NetworkPass {
public:
	using SharedPtr = std::shared_ptr<NetworkServerRecvPass>;

	static SharedPtr create(int texWidth, int texHeight) {
		return SharedPtr(new NetworkServerRecvPass(texWidth, texHeight));
	}
	const NetworkPass::Mode mode = Mode::ServerRecv;

	static std::vector<std::array<float3, 3>> clientCamData;
	static std::array<std::mutex, MAX_NUM_CLIENT> mutexForCamData;
	static std::queue<int> frameNumRendered;

protected:
	NetworkServerRecvPass(int texWidth, int texHeight) 
		: NetworkPass(texWidth, texHeight, "Network Server Recv Pass", "Network Server Recv Pass Options") {}
	void execute(RenderContext* pRenderContext) override;
	bool firstServerRenderUdp(RenderContext* pRenderContext);
	int getClientToRender();

	int clientIndexToRender = 0;

protected:
	bool                                    mFirstRender = true;       ///< If this is the first time rendering, need to send scene
};