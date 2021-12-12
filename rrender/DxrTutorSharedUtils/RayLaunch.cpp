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

#include "RayLaunch.h"

RayLaunch::SharedPtr RayLaunch::RayLaunch::create(uint32_t missCount, uint32_t rayTypeCount, const std::string &rayGenFile, const std::string& rayGenEntryPoint, int recursionDepth, int maxPayloadSize)
{
    return SharedPtr(new RayLaunch(missCount, rayTypeCount, rayGenFile, rayGenEntryPoint, recursionDepth, maxPayloadSize));
}

RayLaunch::RayLaunch(uint32_t missCount, uint32_t rayTypeCount, const std::string &rayGenFile, const std::string& rayGenEntryPoint, int recursionDepth, int maxPayloadSize)
{
    setMaxRecursionDepth(recursionDepth);
    setMaxPayloadSize(maxPayloadSize);
    mRayProgDesc.addShaderLibrary(rayGenFile);
    mpLastShaderFile = rayGenFile;

    mpRayProg = nullptr;
    mpRayVars = nullptr;
    mpScene = nullptr;
    mInvalidVarReflector = true;

    mpRtBindingTable = nullptr;
    mpRayGenEntryPoint = rayGenEntryPoint;
    mMissCount = missCount;
    mRayTypeCount = rayTypeCount;
}

uint32_t RayLaunch::addMissShader(const std::string& missShaderFile, const std::string& missEntryPoint)
{
    if (mpLastShaderFile != missShaderFile)
        mRayProgDesc.addShaderLibrary(missShaderFile);

    mpRtBindingTable->setMiss(mNumMiss, mRayProgDesc.addMiss(missEntryPoint));
    return mNumMiss++;
}

uint32_t RayLaunch::addHitShader(const std::string& hitShaderFile, const std::string& closestHitEntryPoint, const std::string& anyHitEntryPoint)
{
    if (mpLastShaderFile != hitShaderFile)
        mRayProgDesc.addShaderLibrary(hitShaderFile);

    mpRtBindingTable->setHitGroupByType(mNumHitGroup, mpScene, Scene::GeometryType::TriangleMesh, mRayProgDesc.addHitGroup(closestHitEntryPoint, anyHitEntryPoint));
    return mNumHitGroup++;
}

uint32_t RayLaunch::addHitGroup(const std::string& hitShaderFile, const std::string& closestHitEntryPoint, const std::string& anyHitEntryPoint, const std::string& intersectionEntryPoint)
{
    if (mpLastShaderFile != hitShaderFile)
        mRayProgDesc.addShaderLibrary(hitShaderFile);

    mpRtBindingTable->setHitGroupByType(mNumHitGroup, mpScene, Scene::GeometryType::TriangleMesh, mRayProgDesc.addHitGroup(closestHitEntryPoint, anyHitEntryPoint, intersectionEntryPoint));
    return mNumHitGroup++;
}

void RayLaunch::compileRayProgram()
{    
    if (mpScene) {
        // RtPrograms must be created with RtProgram::Desc that references a scene. We make a
        // copy of the stashed Desc (created with the shaders that were added), then add the scene
        // defines, and create the RtProgram. This way, if we set the scene again, we keep the original
        // desc from the shaders.
        auto rayProgDescCopy = mRayProgDesc;
        rayProgDescCopy.addDefines(mpScene->getSceneDefines());

        mpRayProg = RtProgram::create(rayProgDescCopy);
        mInvalidVarReflector = true;

        // Since generating ray tracing variables take a while, try to do it now.
        createRayTracingVariables();
    }
}

bool RayLaunch::readyToRender()
{
    // Do we already know everything is ready?
    if (!mInvalidVarReflector && mpRayProg && mpRayVars) return true;

    // No?  Try creating our variable reflector.
    createRayTracingVariables();

    // See if we're ready now.
    return (mpRayProg && mpRayVars);
}

void RayLaunch::setMaxRecursionDepth(uint32_t maxDepth)
{
    mRayProgDesc.setMaxTraceRecursionDepth(maxDepth);
    mInvalidVarReflector = true;
}

void RayLaunch::setMaxPayloadSize(uint32_t maxPayloadSize)
{
    mRayProgDesc.setMaxPayloadSize(maxPayloadSize);
}

void RayLaunch::setScene(Scene::SharedPtr pScene)
{
    // Make sure we have a valid scene 
    if (!pScene) return;
    mpScene = pScene;

    // Since the scene is an integral part of the variable reflector, we now need to update it!
    mInvalidVarReflector = true;

    // Create RtBindingTable with scene geometry count
    mpRtBindingTable = RtBindingTable::create(mMissCount, mRayTypeCount, mpScene->getGeometryCount());
    mpRtBindingTable->setRayGen(mRayProgDesc.addRayGen(mpRayGenEntryPoint));

    // Since generating our ray tracing variables take quite a while (incl. build time), check if we can do it now
    if (mpRayProg) createRayTracingVariables();
}

void RayLaunch::addDefine(const std::string& name, const std::string& value)
{
    mpRayProg->addDefine(name, value);
    mInvalidVarReflector = true;
}

void RayLaunch::removeDefine(const std::string& name)
{
    mpRayProg->removeDefine(name);
    mInvalidVarReflector = true;
}

void RayLaunch::createRayTracingVariables()
{
    if (mpRayProg && mpScene)
    {
        mpRayVars = RtProgramVars::create(mpRayProg, mpRtBindingTable);
        if (!mpRayVars) return;
        mInvalidVarReflector = false;

        mpRayGenVars = mpRayVars->getRayGenVars();

        mpMissVars.clear();
        for (int i = 0; i<int(mNumMiss); i++)
        {
            mpMissVars.push_back(mpRayVars->getMissVars(i));
        }

        mpHitVars.clear();
        for (int i = 0; i<int(mNumHitGroup); i++)
        {
            int32_t curHitVarsIdx = int32_t(mpHitVars.size());

            GroupVarsVector curVarVec;
            mpHitVars.push_back(curVarVec); 

            for (int j = 0; j<int(mpScene->getMeshCount()); j++)
            {
                auto instanceVar = mpRayVars->getHitVars(i, j);
                mpHitVars[curHitVarsIdx].push_back(instanceVar);
            }
        }
    }
}

RtProgramVars::SharedPtr RayLaunch::getRayVars() {
    if (mInvalidVarReflector) createRayTracingVariables();
    return (!mpRayVars) ? nullptr : mpRayVars;
}

EntryPointGroupVars::SharedPtr RayLaunch::getRayGenVars()
{
    if (mInvalidVarReflector) createRayTracingVariables();
    return (!mpRayVars) ? nullptr : mpRayGenVars;
}

EntryPointGroupVars::SharedPtr RayLaunch::getMissVars(uint32_t rayType)
{
    if (mInvalidVarReflector) createRayTracingVariables();
    return (!mpRayVars || rayType >= uint32_t(mpMissVars.size())) ? nullptr : mpMissVars[rayType];
}

RayLaunch::GroupVarsVector RayLaunch::getHitVars(uint32_t rayType)
{
    if (mInvalidVarReflector) createRayTracingVariables();
    return (!mpRayVars || rayType >= uint32_t(mpHitVars.size())) ? mDefaultHitVarList : mpHitVars[rayType];
}

void RayLaunch::execute(RenderContext::SharedPtr pRenderContext, uint2 rayLaunchDimensions, Camera::SharedPtr viewCamera)
{
    this->execute(pRenderContext.get(), rayLaunchDimensions, viewCamera);
}

void RayLaunch::execute(RenderContext* pRenderContext, uint2 rayLaunchDimensions, Camera::SharedPtr viewCamera)
{
    // Ok.  We're executing.  If we still have an invalid shader variable reflector, we'd better get one now!
    if (mInvalidVarReflector) createRayTracingVariables();

    // We need our shader variable reflector in order to execute!
    if (!mpRayVars) return;

    // Ok.  We're ready and have done all our error checking.  Launch the ray tracing!
    mpScene->raytrace(pRenderContext, mpRayProg.get(), mpRayVars, uint3(rayLaunchDimensions.x, rayLaunchDimensions.y, 1));
}
