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

#include "RasterLaunch.h"


RasterLaunch::SharedPtr RasterLaunch::RasterLaunch::create(Program::Desc& existingDesc)
{
    return SharedPtr(new RasterLaunch(existingDesc));
}

RasterLaunch::SharedPtr RasterLaunch::RasterLaunch::createFromFiles(const std::string& vertexFile, const std::string& fragmentFile)
{
    Program::Desc desc;
    desc.addShaderLibrary(vertexFile).vsEntry("main");
    desc.addShaderLibrary(fragmentFile).psEntry("main");
    return create(desc);
}

RasterLaunch::SharedPtr RasterLaunch::RasterLaunch::createFromFiles(const std::string& vertexFile, const std::string& geometryFile, const std::string& fragmentFile)
{
    Program::Desc desc;
    desc.addShaderLibrary(vertexFile).vsEntry("main");
    desc.addShaderLibrary(geometryFile).gsEntry("main");
    desc.addShaderLibrary(fragmentFile).psEntry("main");
    return create(desc);
}

RasterLaunch::SharedPtr RasterLaunch::RasterLaunch::createFromFiles(const std::string& vertexFile, const std::string& fragmentFile, const std::string& geometryFile, const std::string& hullFile, const std::string& domainFile)
{
    Program::Desc desc;
    desc.addShaderLibrary(vertexFile).vsEntry("main");
    desc.addShaderLibrary(hullFile).hsEntry("main");
    desc.addShaderLibrary(domainFile).dsEntry("main");
    desc.addShaderLibrary(geometryFile).gsEntry("main");
    desc.addShaderLibrary(fragmentFile).psEntry("main");
    return create(desc);
}

RasterLaunch::RasterLaunch(Program::Desc& existingDesc)
{
    mProgDesc = existingDesc;
    mpScene = nullptr;
    mpPassShader = nullptr;
    mpSharedVars = nullptr;
    mInvalidVarReflector = true;
}

void RasterLaunch::addDefine(const std::string& name, const std::string& value)
{
    mpPassShader->addDefine(name, value);
    mInvalidVarReflector = true;
}

void RasterLaunch::removeDefine(const std::string& name)
{
    mpPassShader->removeDefine(name);
    mInvalidVarReflector = true;
}

GraphicsVars::SharedPtr RasterLaunch::getVars()
{
    if (mInvalidVarReflector)
        createGraphicsVariables();

    return mpSharedVars;
}

void RasterLaunch::setScene(Scene::SharedPtr pScene)
{
    if (!pScene)
    {
        mpScene = nullptr;
        return;
    }
    mpPassShader = GraphicsProgram::create(mProgDesc, pScene->getSceneDefines());
    mInvalidVarReflector = true;
    mpScene = pScene;
}

void RasterLaunch::createGraphicsVariables()
{
    // Do we need to recreate our variables?  Do we also have a valid shader?
    if (mInvalidVarReflector && mpPassShader)
    {
        mpSharedVars = GraphicsVars::create(mpPassShader->getActiveVersion()->getReflector());
        mInvalidVarReflector = false;
    }
}

void RasterLaunch::execute(RenderContext::SharedPtr pRenderContext, GraphicsState::SharedPtr pGfxState, const Fbo::SharedPtr &pTargetFbo)
{
    this->execute(pRenderContext.get(), pGfxState, pTargetFbo);
}

void RasterLaunch::execute(RenderContext* pRenderContext, GraphicsState::SharedPtr pGfxState, const Fbo::SharedPtr &pTargetFbo, bool setVp0Sc0)
{
    // Ok.  We're executing.  If we still have an invalid shader variable reflector, we'd better get one now!
    if (mInvalidVarReflector) createGraphicsVariables();

    // If we all the resources we need are valid, go ahead and render
    if (mpPassShader && mpScene && pGfxState && pRenderContext)
    {
        if(pTargetFbo) pGfxState->setFbo(pTargetFbo, setVp0Sc0);
        pGfxState->setProgram(mpPassShader);
        mpScene->rasterize(pRenderContext, pGfxState.get(), mpSharedVars.get());
    }
}
