#pragma once

#include "NetworkPass.h"

class NetworkClientRecvPass : public :: NetworkPass {
public:
	using SharedPtr = std::shared_ptr<NetworkClientRecvPass>;

	static SharedPtr create() {
		return SharedPtr(new NetworkClientRecvPass());
	}
	const NetworkPass::Mode mode = Mode::ClientRecv;

protected:
	NetworkClientRecvPass() : NetworkPass() {}
	void execute(RenderContext* pRenderContext) override;
	
	
	
protected:
	bool                                    firstClientReceive = true; // Use a longer timeout for first client receive
};