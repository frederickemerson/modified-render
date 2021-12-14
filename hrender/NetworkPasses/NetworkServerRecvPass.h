#pragma once

#include "NetworkPass.h"

class NetworkServerRecvPass : public :: NetworkPass {
public:
	using SharedPtr = std::shared_ptr<NetworkServerRecvPass>;

	static SharedPtr create() {
		return SharedPtr(new NetworkServerRecvPass());
	}
	const NetworkPass::Mode mode = Mode::ServerRecv;

protected:
	NetworkServerRecvPass() : NetworkPass() {}
	void execute(RenderContext* pRenderContext) override;
	bool firstServerRenderUdp(RenderContext* pRenderContext);

protected:
	bool                                    mFirstRender = true;       ///< If this is the first time rendering, need to send scene
};