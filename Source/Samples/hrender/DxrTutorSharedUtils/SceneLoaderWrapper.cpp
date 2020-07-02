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

#include "SceneLoaderWrapper.h"

using namespace Falcor;

namespace {
    const FileDialogFilterVec kTextureExtensions = { { "hdr" }, { "png" }, { "jpg" }, { "bmp" } };
};

Scene::SharedPtr loadScene( uint2 currentScreenSize, const char *defaultFilename )
{
    Scene::SharedPtr pScene;

    // If we didn't request a file to load, open a dialog box, asking which scene to load; on failure, return invalid scene
    std::string filename;
    if (!defaultFilename)
    {
        if (!openFileDialog(Scene::kFileExtensionFilters, filename))
            return pScene;
    }
    else
    {
        std::string fullPath;

        // Since we often run in Visual Studio, let's also check the relative paths to the binary directory...
        if (!findFileInDataDirectories(std::string(defaultFilename), fullPath))
            return pScene;

        filename = fullPath;
    }

    // Create a loading bar while loading a scene
    ProgressBar::SharedPtr pBar = ProgressBar::show("Loading Scene", 100);

    // Load a scene
    SceneBuilder::SharedPtr pBuilder = SceneBuilder::create(filename);

    // If we have a valid scene, do some sanity checking; set some defaults
    if (pBuilder)
    {
        // Check to ensure the scene has at least one light.  If not, create a simple directional light
        if (pBuilder->getLightCount() == 0)
        {
            DirectionalLight::SharedPtr pDirLight = DirectionalLight::create();
            pDirLight->setWorldDirection(float3(-0.189f, -0.861f, -0.471f));
            pDirLight->setIntensity(float3(1, 1, 0.985f) * 10.0f);
            pDirLight->setName("DirLight");  // In case we need to display it in a GUI
            pBuilder->addLight(pDirLight);
        }

        // Finalize the SceneBuilder's Scene
        pScene = pBuilder->getScene();

        // Bind a sampler to all scene textures using linear filtering.  Note this is only used if 
        //    you use Falcor's built-in shading system.  Otherwise, you may have to specify your own sampler elsewhere.
        Sampler::Desc desc;
        desc.setFilterMode(Sampler::Filter::Linear, Sampler::Filter::Linear, Sampler::Filter::Linear);
        Sampler::SharedPtr pSampler = Sampler::create(desc);
        pScene->bindSamplerToMaterials(pSampler);

        // Set the aspect ratio of the camera appropriately
        pScene->getCamera()->setAspectRatio((float)currentScreenSize.x / (float)currentScreenSize.y);

        // If scene has a camera animation, disable from starting animation at load.
        pScene->toggleCameraAnimation(false);
    }

    // We're done.  Return whatever scene we might have
    return pScene;
}

std::string getTextureLocation(bool &isValid)
{
    // Open a dialog box, asking which scene to load; on failure, return invalid scene
    std::string filename;
    if (!openFileDialog(kTextureExtensions, filename))
    {
        isValid = false;
        return std::string("");
    }

    isValid = true;
    return filename;
}
