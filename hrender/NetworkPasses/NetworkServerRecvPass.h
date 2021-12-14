#pragma once

#include "NetworkPass.h"

class NetworkServerRecvPass : public :: NetworkPass {
public:
	using SharedPtr = std::shared_ptr<NetworkServerRecvPass>;

	static SharedPtr create(int texWidth, int texHeight) {
		return SharedPtr(new NetworkServerRecvPass(texWidth, texHeight));
	}
	const NetworkPass::Mode mode = Mode::ServerRecv;

protected:
	NetworkServerRecvPass(int texWidth, int texHeight) 
		: NetworkPass(texWidth, texHeight, "Network Server Recv Pass", "Network Server Recv Pass Options") {}
	void execute(RenderContext* pRenderContext) override;
	bool firstServerRenderUdp(RenderContext* pRenderContext);

protected:
	bool                                    mFirstRender = true;       ///< If this is the first time rendering, need to send scene
};