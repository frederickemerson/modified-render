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

// Implementation of the SVGF paper.  For details, please see this paper:
//       http://research.nvidia.com/publication/2017-07_Spatiotemporal-Variance-Guided-Filtering%3A

#include "SVGFServerPass.h"

namespace {
    // Where is our shaders located?
    const char *kReprojectShader         = "Samples\\hrender\\SVGFPasses\\Data\\SVGFPasses\\SVGF1ColorReproject.ps.hlsl";
    const char *kAtrousShader            = "Samples\\hrender\\SVGFPasses\\Data\\SVGFPasses\\SVGFServerAtrous.ps.hlsl";
    const char *kModulateShader          = "Samples\\hrender\\SVGFPasses\\Data\\SVGFPasses\\SVGFServerModulate.ps.hlsl";
    const char *kFilterMomentShader      = "Samples\\hrender\\SVGFPasses\\Data\\SVGFPasses\\SVGF1ColorFilterMoments.ps.hlsl";
};

SVGFServerPass::SharedPtr SVGFServerPass::create(const std::string &indirectIn, const std::string &outChannel, int texWidth, int texHeight)
{
    return SharedPtr(new SVGFServerPass(indirectIn, outChannel, texWidth, texHeight));
}

SVGFServerPass::SVGFServerPass(const std::string &indirectIn, const std::string &outChannel, int texWidth, int texHeight)
    : RenderPass( "Spatiotemporal Filter (SVGF)", "SVGF Options" )
{
    mIndirectInTexName = indirectIn; 
    mOutTexName        = outChannel; // Indirect color written RGB, A contains light index (see ggxServerGlobalIllumination)
    mTexHeight = texHeight;
    mTexWidth = texWidth;
}

bool SVGFServerPass::initialize(RenderContext* pRenderContext, ResourceManager::SharedPtr pResManager)
{
    // Stash a copy of our resource manager so we can get rendering resources
    mpResManager = pResManager;

    // Set our input textures / resources.  These are managed by our resource manager, and are 
    //    created each frame by earlier render passes.
    mpResManager->requestTextureResource("WorldPosition");
    mpResManager->requestTextureResource("__TextureData"); // We need this to retrieve the emissive color
    mpResManager->requestTextureResource(mIndirectInTexName);
    mpResManager->requestTextureResource("SVGF_LinearZ");  
    mpResManager->requestTextureResource("SVGF_MotionVecs");  
    mpResManager->requestTextureResource("SVGF_CompactNormDepth");
    mpResManager->requestTextureResource("OutIndirectAlbedo");

    // Set the output channel
    mpResManager->requestTextureResource("IntermediateOutput", ResourceFormat::RGBA32Float, ResourceManager::kDefaultFlags, mTexWidth, mTexHeight); // RGBA32Float version of output
    mpResManager->requestTextureResource(mOutTexName, ResourceFormat::RGBA8Uint, ResourceManager::kDefaultFlags, mTexWidth, mTexHeight);

    // Create our graphics state
    mpSvgfState = GraphicsState::create();

    // Setup our filter shaders
    mpReprojection      = FullscreenLaunch::create(kReprojectShader);
    mpAtrous            = FullscreenLaunch::create(kAtrousShader);
    mpModulate          = FullscreenLaunch::create(kModulateShader);
    mpFilterMoments     = FullscreenLaunch::create(kFilterMomentShader);

    // Our GUI needs more space than other passes, so enlarge the GUI window.
    setGuiSize(int2(300, 350));

    return true;
}

void SVGFServerPass::resize(uint32_t width, uint32_t height)
{
    // Skip if we're resizing to 0 width or height.
    if (width <= 0 || height <= 0) return;

    // Have 3 different types of framebuffers and resources.  Reallocate them whenever screen resolution changes.

    {   // Type 1, Screen-size FBOs with 1 RGBA32F MRT1
        Fbo::Desc desc;
        desc.setSampleCount(0);
        desc.setColorTarget(0, ResourceFormat::RGBA32Float);
        mpPingPongFbo[0]  = Fbo::create2D(width, height, desc);
        mpPingPongFbo[1] = Fbo::create2D(width, height, desc);
        mpFilteredPastFbo = Fbo::create2D(width, height, desc);
    }

    {   // Type 2, Screen-size FBOs with 4 MRTs, 3 that are RGBA32F, one that is R16F
        Fbo::Desc desc;
        desc.setSampleCount(0);
        desc.setColorTarget(0, Falcor::ResourceFormat::RGBA32Float); // indirect
        desc.setColorTarget(1, Falcor::ResourceFormat::RGBA32Float); // moments
        desc.setColorTarget(2, Falcor::ResourceFormat::R16Float);    // history length
        mpCurReprojFbo = Fbo::create2D(width, height, desc);
        mpPrevReprojFbo = Fbo::create2D(width, height, desc);
    }

    {   // Type 3, Screen-size FBOs with 1 RGBA32F buffer
        Fbo::Desc desc;
        desc.setColorTarget(0, Falcor::ResourceFormat::RGBA32Float); 
        mpOutputFbo = Fbo::create2D(width, height, desc);
    }
    
    // We're manually keeping a copy of our linear Z G-buffers from frame N for use in rendering frame N+1
    mInputTex.prevLinearZ = Texture::create2D(width, height, ResourceFormat::RGBA32Float, 1, 1, nullptr, Resource::BindFlags::ShaderResource | Resource::BindFlags::UnorderedAccess | Resource::BindFlags::RenderTarget);

    mNeedFboClear = true;
}

void SVGFServerPass::clearFbos(RenderContext* pCtx)
{
    // Clear our FBOs
    pCtx->clearFbo(mpPrevReprojFbo.get(), float4(0), 1.0f, 0, FboAttachmentType::All);
    pCtx->clearFbo(mpCurReprojFbo.get(), float4(0), 1.0f, 0, FboAttachmentType::All);
    pCtx->clearFbo(mpFilteredPastFbo.get(), float4(0), 1.0f, 0, FboAttachmentType::All);
    pCtx->clearFbo(mpPingPongFbo[0].get(), float4(0), 1.0f, 0, FboAttachmentType::All);
    //pCtx->clearFbo(mpPingPongFbo[1].get(), float4(0), 1.0f, 0, FboAttachmentType::All);

    // Clear our history textures
    pCtx->clearUAV(mInputTex.prevLinearZ->getUAV().get(), float4(0.f, 0.f, 0.f, 1.f));

    mNeedFboClear = false;
}

void SVGFServerPass::renderGui(Gui::Window* pPassWindow)
{
    // Commented out GUI fields don't currently work with the current SVGF implementation
    int dirty = 0;
    dirty |= (int)pPassWindow->checkbox(mFilterEnabled ? "SVGF enabled" : "SVGF disabled", mFilterEnabled);

    pPassWindow->text("");
    pPassWindow->text("Number of filter iterations.  Which");
    pPassWindow->text("    iteration feeds into future frames?");
    dirty |= (int)pPassWindow->var("Iterations", mFilterIterations, 2, 10, 0.2f);
    dirty |= (int)pPassWindow->var("Feedback", mFeedbackTap, -1, mFilterIterations-2, 0.2f);

    pPassWindow->text("");
    pPassWindow->text("Contol edge stopping on bilateral filter");
    dirty |= (int)pPassWindow->var("For Color", mPhiColor, 0.0f, 10000.0f, 0.05f);
    dirty |= (int)pPassWindow->var("For Normal", mPhiNormal, 0.001f, 10000.0f, 0.5f);

    pPassWindow->text("");
    pPassWindow->text("How much history should be used?");
    pPassWindow->text("    (alpha; 0 = full reuse; 1 = no reuse)");
    dirty |= (int)pPassWindow->var("Alpha", mAlpha, 0.0f, 1.0f, 0.01f);
    dirty |= (int)pPassWindow->var("Moments Alpha", mMomentsAlpha, 0.0f, 1.0f, 0.01f);

    if (dirty)
    {
        // Flag to the renderer that options that affect the rendering have changed.
        setRefreshFlag();
    }
}

void SVGFServerPass::execute(RenderContext* pRenderContext)
{
    // Ensure we have received information about our rendering state, or we can't render.
    if (!mpResManager) return;

    // Grab our output texture  Make sure it exists
    Texture::SharedPtr pDst = mpResManager->getTexture(mOutTexName);
    if (!pDst) return;

    // Do we need to clear our internal framebuffers?  If so, do it.
    if (mNeedFboClear) clearFbos(pRenderContext);

    // Set up our textures to point appropriately
    mInputTex.indirectIllum = mpResManager->getTexture(mIndirectInTexName);
    mInputTex.linearZ       = mpResManager->getTexture("SVGF_LinearZ");
    mInputTex.motionVecs    = mpResManager->getTexture("SVGF_MotionVecs");
    mInputTex.miscBuf       = mpResManager->getTexture("SVGF_CompactNormDepth");
    mInputTex.indirAlbedo   = mpResManager->getTexture("OutIndirectAlbedo");


    if (mFilterEnabled)
    {
        // Perform the major passes in SVGF filtering
        computeReprojection(pRenderContext);
        computeVarianceEstimate(pRenderContext);
        computeAtrousDecomposition(pRenderContext);

        // This performs the modulation in case there are no wavelet iterations performed
        if (mFilterIterations <= 0)
            computeModulation(pRenderContext);

        // Output the result of SVGF to the expected output buffer for subsequent passes.
        // For the server, the color is written to the output texture directly, as we need to convert the format.
        //pRenderContext->blit(mpOutputFbo->getColorTexture(0)->getSRV(), pDst->getRTV());

        // Swap resources so we're ready for next frame.
        std::swap(mpCurReprojFbo, mpPrevReprojFbo);
        pRenderContext->blit(mInputTex.linearZ->getSRV(), mInputTex.prevLinearZ->getRTV());
    }
    else
    {
        // No SVGF.  Just perform modulation
        computeModulation(pRenderContext);
    }
}



void SVGFServerPass::computeReprojection(RenderContext* pRenderContext)
{
    auto reprojVars = mpReprojection->getVars();
    // Setup textures for our reprojection shader pass
    reprojVars["gLinearZ"]       = mInputTex.linearZ;
    reprojVars["gPrevLinearZ"]   = mInputTex.prevLinearZ;
    reprojVars["gMotion"]        = mInputTex.motionVecs;
    reprojVars["gPrevColor"] = mpFilteredPastFbo->getColorTexture(0);
    reprojVars["gPrevMoments"]   = mpPrevReprojFbo->getColorTexture(1);
    reprojVars["gHistoryLength"] = mpPrevReprojFbo->getColorTexture(2);
    reprojVars["gColor"]      = mInputTex.indirectIllum;

    // Setup variables for our reprojection pass
    reprojVars["PerImageCB"]["gAlpha"] = mAlpha;
    reprojVars["PerImageCB"]["gMomentsAlpha"] = mMomentsAlpha;

    // Execute the reprojection pass
    mpReprojection->execute(pRenderContext, mpCurReprojFbo);
}

void SVGFServerPass::computeVarianceEstimate(RenderContext* pRenderContext)
{
    auto filterMomentsVars = mpFilterMoments->getVars();
    filterMomentsVars["gColor"] = mpCurReprojFbo->getColorTexture(0);
    filterMomentsVars["gMoments"] = mpCurReprojFbo->getColorTexture(1);
    filterMomentsVars["gHistoryLength"] = mpCurReprojFbo->getColorTexture(2);
    filterMomentsVars["gCompactNormDepth"] = mInputTex.miscBuf;

    filterMomentsVars["PerImageCB"]["gPhiColor"] = mPhiColor;
    filterMomentsVars["PerImageCB"]["gPhiNormal"] = mPhiNormal;

    mpFilterMoments->execute(pRenderContext, mpPingPongFbo[0]);
}

void SVGFServerPass::computeAtrousDecomposition(RenderContext* pRenderContext)
{
    auto atrousVars = mpAtrous->getVars();
    atrousVars["PerImageCB"]["gPhiColor"]  = mPhiColor;
    atrousVars["PerImageCB"]["gPhiNormal"] = mPhiNormal;
    atrousVars["gHistoryLength"]           = mpCurReprojFbo->getColorTexture(3);
    atrousVars["gCompactNormDepth"]        = mInputTex.miscBuf;
    // Used for emissive calculation in the last filter iteration which performs modulation
    //atrousVars["gTexData"] = mpResManager->getTexture("__TextureData");
    //atrousVars["PerImageCB"]["gEmitMult"] = 1.0f;

    for (int i = 0; i < mFilterIterations; i++) {
        bool performModulation = (i == mFilterIterations - 1);
        Fbo::SharedPtr curTargetFbo = performModulation ? mpOutputFbo : mpPingPongFbo[1]; 

        // Send down our input images
        atrousVars["gColor"] = mpPingPongFbo[0]->getColorTexture(0);   
        atrousVars["PerImageCB"]["gStepSize"] = 1 << i;

        // perform modulation in-shader if needed
        atrousVars["PerImageCB"]["gPerformModulation"] = performModulation;
        atrousVars["gAlbedo"] = mInputTex.indirAlbedo;

        // Output texture we write to
        atrousVars["gOutput"] = mpResManager->getTexture(mOutTexName);

        mpAtrous->execute(pRenderContext, curTargetFbo);

        // store the filtered color for the feedback path
        if (i == std::min(mFeedbackTap, mFilterIterations - 1))
        {
            pRenderContext->blit(curTargetFbo->getColorTexture(0)->getSRV(), mpFilteredPastFbo->getRenderTargetView(0));
        }

        std::swap(mpPingPongFbo[0], mpPingPongFbo[1]); 
    }

    if (mFeedbackTap < 0 || mFilterIterations <= 0)
    {
        pRenderContext->blit(mpCurReprojFbo->getColorTexture(0)->getSRV(), mpFilteredPastFbo->getRenderTargetView(0));
    }

}

void SVGFServerPass::computeModulation(RenderContext* pRenderContext)
{
    // We just need to leave the input data unchanged.
    auto modulateVars = mpModulate->getVars();
    // If filtering is enabled, we use modulate with the illumination from the new frame integrated with history,
    // otherwise we use the new illumination frame directly.
    modulateVars["gIndirect"] = mFilterEnabled ? mpCurReprojFbo->getColorTexture(0) : mInputTex.indirectIllum;
    modulateVars["gIndirAlbedo"] = mInputTex.indirAlbedo;

    modulateVars["gOutput"] = mpResManager->getTexture(mOutTexName);

    // Run the modulation pass. If filtering is not enabled, we simply render to the output texture. Otherwise,
    // we render to the output fbo, since there is further logic to perform (copying to history for the next frame)
    mpModulate->execute(pRenderContext, mFilterEnabled ? mpOutputFbo : mpResManager->createManagedFbo({ "IntermediateOutput" }));
}

