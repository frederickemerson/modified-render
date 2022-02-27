#pragma once

#include "NetworkPass.h"

class NetworkClientRecvWaitingPass : public :: NetworkPass {
public:
	using SharedPtr = std::shared_ptr<NetworkClientRecvWaitingPass>;

	static SharedPtr create(int texWidth, int texHeight) {
		return SharedPtr(new NetworkClientRecvWaitingPass(texWidth, texHeight));
	}
	const NetworkPass::Mode mode = Mode::ClientRecv;

protected:
	NetworkClientRecvWaitingPass(int texWidth, int texHeight)
		: NetworkPass(texWidth, texHeight, "Network Client Recv Waiting Pass", "Network Client Recv Waiting Pass Options") {}

	void execute(RenderContext* pRenderContext) override;
	
protected:
	bool                                    firstClientReceive = true; // Use a longer timeout for first client receive
};