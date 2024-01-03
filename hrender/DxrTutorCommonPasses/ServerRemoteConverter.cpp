#include "ServerRemoteConverter.h"

using namespace std;

namespace {
    // // Where is our environment map and scene located?
    // const char* kEnvironmentMap = "MonValley_G_DirtRoad_3k.hdr";
    // const char* kDefaultScene = "pink_room\\pink_room.pyscene";
    // What are the entry points in that shader for various ray tracing shaders?
    const char* kEntryPointRayGen = "ColorsRayGen"; 

    // Where is our color conversion shader located?
    const char* kFileColorConvert = "Samples\\hrender\\DxrTutorCommonPasses\\Data\\CommonPasses\\floatConvert.hlsl";
};

bool ServerRemoteConverter::initialize(RenderContext* pRenderContext, ResourceManager::SharedPtr pResManager)
{
    // Stash a copy of our resource manager so we can get rendering resources
    mpResManager = pResManager;

    // Our GUI needs less space than other passes, so shrink the GUI window.
    setGuiSize(Falcor::int2(300, 70));

    // Note that we some buffers from the G-buffer, plus the standard output buffer
    mInputIndex = mpResManager->requestTextureResource(mInputTexName);
    mOutputIndex = mpResManager->requestTextureResource(mOutputTexName, ResourceFormat::R11G11B10Float, ResourceManager::kDefaultFlags, mTexWidth, mTexHeight);

    // Create our wrapper around a ray tracing pass.  Tell it where our ray generation shader and ray-specific shaders are
    mpRays = RayLaunch::create(0, 1, kFileColorConvert, kEntryPointRayGen);

    // Now that we've passed all our shaders in, compile and (if available) setup the scene
    if (mpScene) {
        mpRays->setScene(mpScene);
        mpRays->compileRayProgram();
    }

    return true;
}

void ServerRemoteConverter::initScene(RenderContext* pRenderContext, Scene::SharedPtr pScene)
{
    // Stash a copy of the scene and pass it to our ray tracer (if initialized)
    mpScene = pScene;
    if (!mpScene) return;

    // Create our wrapper around a ray tracing pass.  Tell it where our ray generation shader and ray-specific shaders are
    mpRays = RayLaunch::create(0, 1, kFileColorConvert, kEntryPointRayGen);

    if (mpRays) {
        mpRays->setScene(mpScene);
        mpRays->compileRayProgram();
    }
}

void ServerRemoteConverter::execute(RenderContext* pRenderContext)
{
    // Get the output buffer we're writing into
    //Texture::SharedPtr pDstTex = mpResManager->getTexture(mOutputIndex);
    Texture::SharedPtr pSrcTex = mpResManager->getTexture(mInputIndex);
    Texture::SharedPtr pDstTex = mpResManager->getTexture(mOutputIndex);

    // Do we have all the resources we need to render?  If not, return
    if (!pDstTex || !pSrcTex || !mpRays || !mpRays->readyToRender()) {
        OutputDebugString(L"\n\nUnable to color convert in ServerRemoteConverter.\n");
        return;
    }

    // Set our ray tracing shader variables
    auto rayVars = mpRays->getRayVars();
    rayVars["gCompact"] = pDstTex;
    rayVars["gUncompact"] = pSrcTex;
    rayVars["PerImageCB"]["gIsCompacting"] = true;

    // Shoot our rays and shade our primary hit points
    mpRays->execute(pRenderContext, Falcor::uint2(pDstTex->getWidth(), pDstTex->getHeight()));
}

void ServerRemoteConverter::renderGui(Gui::Window* pPassWindow)
{
    int dirty = 0;

    // If any of our UI parameters changed, let the pipeline know we're doing something different next frame
    if (dirty) setRefreshFlag();
}
