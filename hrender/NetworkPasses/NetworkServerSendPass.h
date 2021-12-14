#pragma once

#include "NetworkPass.h"

class NetworkServerSendPass : public :: NetworkPass {
public:
	using SharedPtr = std::shared_ptr<NetworkServerSendPass>;

	static SharedPtr create(int texWidth, int texHeight) {
		return SharedPtr(new NetworkServerSendPass(texWidth, texHeight));
	}
	const NetworkPass::Mode mode = Mode::ServerSend;


protected:
	NetworkServerSendPass(int texWidth, int texHeight) 
		:NetworkPass(texWidth, texHeight, "Network Server Send Pass", "Network Server Send Pass Options") {
		mTexWidth = texWidth;
		mTexHeight = texHeight;
	}	
	void execute(RenderContext* pRenderContext) override;

protected:
	bool                                    firstServerSend = true; // Set off the server network thread on first send
	int                                     mTexWidth = -1;            ///< The width of the texture we render, based on the client
	int                                     mTexHeight = -1;           ///< The height of the texture we render, based on the client
};