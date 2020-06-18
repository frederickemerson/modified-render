/***************************************************************************
 # Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
 #
 # Redistribution and use in source and binary forms, with or without
 # modification, are permitted provided that the following conditions
 # are met:
 #  * Redistributions of source code must retain the above copyright
 #    notice, this list of conditions and the following disclaimer.
 #  * Redistributions in binary form must reproduce the above copyright
 #    notice, this list of conditions and the following disclaimer in the
 #    documentation and/or other materials provided with the distribution.
 #  * Neither the name of NVIDIA CORPORATION nor the names of its
 #    contributors may be used to endorse or promote products derived
 #    from this software without specific prior written permission.
 #
 # THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 # EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 # IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 # PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 # CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 # EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 # PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 # PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 # OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 # (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 # OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 **************************************************************************/
#include "hrender.h"
#include "DxrTutorSharedUtils/RenderingPipeline.h"
#include "DxrTutorTestPasses/SinusoidRasterPass.h"
#include "DxrTutorTestPasses/ConstantColorPass.h"

uint32_t mSampleGuiWidth = 250;
uint32_t mSampleGuiHeight = 200;
uint32_t mSampleGuiPositionX = 20;
uint32_t mSampleGuiPositionY = 40;

void hrender::onGuiRender(Gui* pGui)
{
    Gui::Window w(pGui, "Falcor", { 250, 200 });
    gpFramework->renderGlobalUI(pGui);
    w.text("Hello from hrender");
    if (w.button("Click Here"))
    {
        msgBox("Now why would you do that?");
    }
}

void hrender::onLoad(RenderContext* pRenderContext)
{
}

void hrender::onFrameRender(RenderContext* pRenderContext, const Fbo::SharedPtr& pTargetFbo)
{
    const float4 clearColor(0.38f, 0.52f, 0.10f, 1);
    pRenderContext->clearFbo(pTargetFbo.get(), clearColor, 1.0f, 0, FboAttachmentType::All);
}

void hrender::onShutdown()
{
}

bool hrender::onKeyEvent(const KeyboardEvent& keyEvent)
{
    return false;
}

bool hrender::onMouseEvent(const MouseEvent& mouseEvent)
{
    return false;
}

void hrender::onHotReload(HotReloadFlags reloaded)
{
}

void hrender::onResizeSwapChain(uint32_t width, uint32_t height)
{
}

int WINAPI WinMain(_In_ HINSTANCE hInstance, _In_opt_ HINSTANCE hPrevInstance, _In_ LPSTR lpCmdLine, _In_ int nShowCmd)
{
    // Create our rendering pipeline
    RenderingPipeline* pipeline = new RenderingPipeline();

    // Add passes into our pipeline
    pipeline->setPass(0, SinusoidRasterPass::create());   // This pass displays a time-varying sinusoidal function
    pipeline->setPass(1, ConstantColorPass::create());   // Displays a user-selectable color on the screen

    // Define a set of config / window parameters for our program
    SampleConfig config;
    config.windowDesc.title = "Tutorial 2:  Running a simple raster pass to generate some more interesting imagry";
    config.windowDesc.resizableWindow = true;

    // Start our program!
    RenderingPipeline::run(pipeline, config);
    return 0;
}
