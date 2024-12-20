/**********************************************************************************************************************
# Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without modification, are permitted provided that the
# following conditions are met:
#  * Redistributions of code must retain the copyright notice, this list of conditions and the following disclaimer.
#  * Neither the name of NVIDIA CORPORATION nor the names of its contributors may be used to endorse or promote products
#    derived from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.  IN NO EVENT
# SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA,
# OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
# LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
# ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
**********************************************************************************************************************/

#include "FullscreenLaunch.h"

using namespace Falcor;

FullscreenLaunch::SharedPtr FullscreenLaunch::FullscreenLaunch::create(const char *fragShader)
{
    return SharedPtr(new FullscreenLaunch(fragShader));
}

FullscreenLaunch::FullscreenLaunch(const char *fragShader)
{ 
    mpPass = FullScreenPass::create(fragShader);
}

void FullscreenLaunch::execute(RenderContext::SharedPtr pRenderContext, Fbo::SharedPtr pTargetFbo)
{
    this->execute(pRenderContext.get(), pTargetFbo);
}

void FullscreenLaunch::execute(RenderContext* pRenderContext, Fbo::SharedPtr pTargetFbo)
{
    if (mpPass && pTargetFbo && pRenderContext)
    {
        mpPass->execute(pRenderContext, pTargetFbo);
    }
}

Falcor::GraphicsVars::SharedPtr FullscreenLaunch::getVars()
{
    return mpPass->getVars();
}

void FullscreenLaunch::addDefine(const std::string& name, const std::string& value)
{
    mpPass->addDefine(name, value, true);
}

void FullscreenLaunch::removeDefine(const std::string& name)
{
    mpPass->removeDefine(name, true);
}
