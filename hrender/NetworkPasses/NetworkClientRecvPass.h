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
		: NetworkPass(texWidth, texHeight, "Network Client Recv Pass", "Network Client Recv Pass Options") {}

	void execute(RenderContext* pRenderContext) override;
	
protected:
	bool                                    firstClientReceive = true; // Use a longer timeout for first client receive
};