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

#include "VisibilityPass.h"
#include "lz4.h"
#include <chrono>
#include <iostream>
using namespace std::chrono;
using namespace std;

// for nvcomp compression
//#include "compression.h"

namespace {
    // Where is our environment map and scene located?
    //const char* kEnvironmentMap = "MonValley_G_DirtRoad_3k.hdr";
    const char* kDefaultScene = "pink_room\\pink_room.fscene";

    // Where is our shaders located?
    const char* kFileRayTrace = "Samples\\hrender\\NetworkPasses\\Data\\NetworkPasses\\visibilityPass.rt.hlsl";

    // What are the entry points in that shader for various ray tracing shaders?
    const char* kEntryPointRayGen = "SimpleShadowsRayGen";
    const char* kEntryPointMiss0 = "ShadowMiss";
    const char* kEntryAoAnyHit = "ShadowAnyHit";
    const char* kEntryAoClosestHit = "ShadowClosestHit";
};

bool VisibilityPass::initialize(RenderContext* pRenderContext, ResourceManager::SharedPtr pResManager)
{
    // Stash a copy of our resource manager so we can get rendering resources
    mpResManager = pResManager;

    // Our GUI needs less space than other passes, so shrink the GUI window.
    setGuiSize(Falcor::int2(300, 70));

    // Note that we some buffers from the G-buffer, plus the standard output buffer
    mpResManager->requestTextureResource(mPosBufName, ResourceFormat::RGBA32Float, ResourceManager::kDefaultFlags, mTexWidth, mTexHeight);
    mOutputIndex = mpResManager->requestTextureResource(mOutputTexName, ResourceFormat::R32Uint, ResourceManager::kDefaultFlags, mTexWidth, mTexHeight);

    // Set default environment map and scene
    //mpResManager->updateEnvironmentMap(kEnvironmentMap);
    mpResManager->setDefaultSceneName(kDefaultScene);

    // Create our wrapper around a ray tracing pass.  Tell it where our ray generation shader and ray-specific shaders are
    mpRays = RayLaunch::create(kFileRayTrace, kEntryPointRayGen);
    mpRays->addMissShader(kFileRayTrace, kEntryPointMiss0);
    mpRays->addHitShader(kFileRayTrace, kEntryAoClosestHit, kEntryAoAnyHit);

    // Now that we've passed all our shaders in, compile and (if available) setup the scene
    if (mpScene) {
        mpRays->setScene(mpScene);
        mpRays->compileRayProgram();
    }

    // initialisation for compression
    dstData = (char*)malloc(8847360);
    srcData2 = (char*)malloc(8847360);
    //state = malloc(LZ4_sizeofState());

    return true;
}

void VisibilityPass::initScene(RenderContext* pRenderContext, Scene::SharedPtr pScene)
{
    // Stash a copy of the scene and pass it to our ray tracer (if initialized)
    mpScene = pScene;
    if (!mpScene) return;
    if (mpRays) {
        mpRays->setScene(mpScene);
        mpRays->compileRayProgram();
    }
}

void VisibilityPass::execute(RenderContext* pRenderContext)
{
    // Get the output buffer we're writing into
    Texture::SharedPtr pDstTex = mpResManager->getClearedTexture(mOutputIndex, Falcor::float4(0.0f));

    // Do we have all the resources we need to render?  If not, return
    if (!pDstTex || !mpRays || !mpRays->readyToRender()) return;

    // Set our ray tracing shader variables
    auto rayVars = mpRays->getRayVars();
    rayVars["RayGenCB"]["gMinT"] = mpResManager->getMinTDist();
    rayVars["RayGenCB"]["gSkipShadows"] = mSkipShadows;
    rayVars["gPos"] = mpResManager->getTexture(mPosBufName);
    rayVars["gOutput"] = pDstTex;

    // Shoot our rays and shade our primary hit points
    mpRays->execute(pRenderContext, Falcor::uint2(pDstTex->getWidth(), pDstTex->getHeight()));

    /* LZ4 compression
    // 1. set srcSize once only
    if (srcSize == 0) {
        srcSize = static_cast<int>(pDstTex->getTextureSizeInBytes());
        visibilityData = std::vector<uint8_t>(srcSize, 0);
        //srcData = (char*)pDstTex->getTextureData3(pRenderContext, 0, 0, &visibilityData);
    }
    // needed for lzo compression
    //std::vector<char> wrkmem(LZO1X_1_MEM_COMPRESS, 0); 

    //increment counter, every 60 counts prints statistics
    counter += 1;

    // 2. GPU-CPU trsf
    auto start = high_resolution_clock::now();
    //pDstTex->sync();
    srcData = (char*)pDstTex->getTextureData2(pRenderContext, 0, 0, &visibilityData);
    //srcData = (char*)&(visibilityData)[0];
    auto stop = high_resolution_clock::now();
    gpucpu_duration += duration_cast<microseconds>(stop - start).count();

    // 3. Compress
    start = high_resolution_clock::now();
    int dstSize = LZ4_compress_default(srcData, dstData, 8294400, 8847360);
    //lzo_uint dstSize;
    //lzo1x_1_compress((unsigned char*)srcData, srcSize, (unsigned char*)dstData, &dstSize, &wrkmem[0]);

    stop = high_resolution_clock::now();
    compress_duration += duration_cast<microseconds>(stop - start).count();
    compressed_size += dstSize;
    //compressed_size += (int)dstSize;

    // 4. Decompress
    start = high_resolution_clock::now();
    int srcSize2 = LZ4_decompress_safe(dstData, srcData2, dstSize, 8847360);
    //lzo_uint srcSize2;
    //lzo1x_decompress((unsigned char*)dstData, dstSize, (unsigned char*)srcData2, &srcSize2, NULL);

    stop = high_resolution_clock::now();
    decompress_duration += duration_cast<microseconds>(stop - start).count();

    // 5. CPU-GPU trsf
    start = high_resolution_clock::now();

    pDstTex->apiInitPub(srcData2, true);
    stop = high_resolution_clock::now();
    cpugpu_duration += duration_cast<microseconds>(stop - start).count();

    if (counter % frequency == 0) {
        // print statistics every {frequency} frames
        printToDebugWindow("\nGPU-CPU duration: " + to_string(gpucpu_duration / frequency));
        printToDebugWindow("\nCompress: " + to_string(compress_duration / frequency));
        printToDebugWindow("\nDecompress: " + to_string(decompress_duration / frequency));
        printToDebugWindow("\nCPU-GPU duration: " + to_string(cpugpu_duration / frequency));
        printToDebugWindow("\nCompressed size:" + std::to_string(compressed_size / frequency) + "\n");
        // reset to 0;
        gpucpu_duration = 0;
        cpugpu_duration = 0;
        compress_duration = 0;
        decompress_duration = 0;
        compressed_size = 0;
    }
    */

    /*
    // nvcomp GPU compression - currently cant build/install
    size_t uncompressed_bytes = pDstTex->getTextureSizeInBytes();
    Texture::SharedPtr visTex = mpResManager->getTexture("VisibilityBitmap");
    std::vector<uint8_t> uncompressed_data = std::vector<uint8_t>(uncompressed_bytes, 0);
    uncompressed_data = visTex->getTextureData(pRenderContext, 0, 0, &uncompressed_data);

    printToDebugWindow(std::to_string(uncompressed_bytes)+"\n");

    PlsWork::compress(&uncompressed_bytes, uncompressed_data);
    printToDebugWindow(std::to_string(uncompressed_bytes) + "\n");
    */
}

void VisibilityPass::renderGui(Gui::Window* pPassWindow)
{
    int dirty = 0;

    dirty |= (int)pPassWindow->checkbox("Skip shadow computation", mSkipShadows, false);

    // If any of our UI parameters changed, let the pipeline know we're doing something different next frame
    if (dirty) setRefreshFlag();
}
