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

#include "Falcor.h"
#include "RenderingPipeline.h"
#include "SceneLoaderWrapper.h"
#include <algorithm>

namespace {
    const char     *kNullPassDescriptor = "< None >";   ///< Name used in dropdown lists when no pass is selected.
    const char     *kNullPresetDescriptor = "< No preset selected >"; ///< Name used in dropdown lists when no preset is selected.
    const uint32_t  kNullPassId = 0xFFFFFFFFu;          ///< Id used to represent the null pass (using -1).
    const uint32_t  kNullPresetId = 0xFFFFFFFFu;        ///< Id used to represent the null preset (using -1).
};


RenderingPipeline::RenderingPipeline() 
    : IRenderer()
{
}

uint32_t RenderingPipeline::addPass(::RenderPass::SharedPtr pNewPass)
{
    size_t id = mAvailPasses.size();
    mAvailPasses.push_back(pNewPass);
    return uint32_t(id);
}

void RenderingPipeline::setPresets(const std::vector<PresetData>& presets)
{
    // Ensure our lists are empty
    mPresetSelector.resize(0);
    mPresetsData.resize(0);

    // Add an item to allow selecting a null preset.
    mPresetSelector.push_back({ kNullPresetId, kNullPresetDescriptor });

    // Include each preset in the dropdown
    {
        uint32_t i = 0;
        for (auto& preset : presets)
        {
            // Don't add if this preset doesn't have the correct number of passes or has invalid options
            if (!isPresetValid(preset.selectedPassIdxs)) continue;

            // Ok.  The preset is valid. Insert it into the dropdown list and the data list.
            mPresetSelector.push_back({ i++, preset.descriptor });
            mPresetsData.push_back(preset);
        }
    }
}

bool RenderingPipeline::isPresetValid(const std::vector<uint32_t>& presetSequence)
{
    if (presetSequence.size() != mPassSelectors.size()) return false;

    for (uint32_t i = 0; i < presetSequence.size(); i++)
    {
        // If the pass number has explicitly specified pass options, use that as the limit.
        // Otherwise, just use the available passes as the limit (+1 for null pass).
        uint32_t numPasses = mPassSelectors[i].size() != 0 ? uint32_t(mPassSelectors[i].size()) : (uint32_t(mAvailPasses.size()) + 1);
        if (presetSequence[i] >= numPasses) return false;
    }
    return true;
}

void RenderingPipeline::onLoad(RenderContext* pRenderContext)
{
    // Create our resource manager
    mpResourceManager = ResourceManager::create(mLastKnownSize.x, mLastKnownSize.y);
    mOutputBufferIndex = mpResourceManager->requestTextureResource(ResourceManager::kOutputChannel);

    // Initialize all of the RenderPasses we have available to select for our pipeline
    for (uint32_t i = 0; i < mAvailPasses.size(); i++)
    {
        if (mAvailPasses[i])
        {
            // Initialize.  If failure, remove this pass from the list.
            bool initialized = mAvailPasses[i]->onInitialize(pRenderContext, mpResourceManager);
            if (!initialized) mAvailPasses[i] = nullptr;
        }
    }

    // If nobody has started inserting passes into our pipeline, set up our GUI so we can start adding passes manually.
    if (mActivePasses.size() == 0)
    {
        insertPassIntoPipeline(0);
    }

    // Initialize GUI pass selection dropdown lists.
    for (uint32_t i = 0; i < mPassSelectors.size(); i++)
    {
        if (mPassSelectors[i].size() <= 0)
            createDefaultDropdownGuiForPass(i, mPassSelectors[i]);
    }

    // Create identifiers for profiling.
    mProfileGPUTimes.resize(mActivePasses.size() * 2);
    mProfileLastGPUTimes.resize(mActivePasses.size() * 2);
    for (uint32_t i = 0; i < mActivePasses.size()*2; i++)
    {
        char buf[256];
        sprintf_s(buf, "Pass_%d", i);
        mProfileNames.push_back( std::string(buf) );
    }

    // We're going to create a default graphics state, and stash a reference in the resource manager
    mpDefaultGfxState = GraphicsState::create();
    mpResourceManager->setDefaultGfxState(mpDefaultGfxState);

    // If we've requested to have an environment map... 
    if (anyPassUsesEnvMap())
    {
        populateEnvMapSelector();
    }

    // By default, we freeze animation on load
    gpFramework->getGlobalClock().stop();

    // When we initialize, we have a new pipe, so we need to give data to the passes
    updatePipelineRequirementFlags();
    mPipelineChanged = true;

    // Note that we've done our initialzation pass
    mIsInitialized = true;

    return;
}

void RenderingPipeline::updatePipelineRequirementFlags(void)
{
    // Start by changing all flags to false
    mPipeRequiresScene = false;
    mPipeRequiresRaster = false;
    mPipeRequiresRayTracing = false;
    mPipeAppliesPostprocess = false;
    mPipeUsesCompute = false;
    mPipeUsesEnvMap = false;
    mPipeNeedsDefaultScene = mpResourceManager ? mpResourceManager->userSetDefaultScene() : false;
    mPipeHasAnimation = false;

    for (uint32_t passNum = 0; passNum < mActivePasses.size(); passNum++)
    {
        if (mActivePasses[passNum])
        {
            mPipeRequiresScene = mPipeRequiresScene || mActivePasses[passNum]->requiresScene();
            mPipeRequiresRaster = mPipeRequiresRaster || mActivePasses[passNum]->usesRasterization();
            mPipeRequiresRayTracing = mPipeRequiresRayTracing || mActivePasses[passNum]->usesRayTracing();
            mPipeAppliesPostprocess = mPipeAppliesPostprocess || mActivePasses[passNum]->appliesPostprocess();
            mPipeUsesCompute = mPipeUsesCompute || mActivePasses[passNum]->usesCompute();
            mPipeUsesEnvMap = mPipeUsesEnvMap || mActivePasses[passNum]->usesEnvironmentMap();
            mPipeNeedsDefaultScene = mPipeNeedsDefaultScene || mActivePasses[passNum]->loadDefaultScene();
            mPipeHasAnimation = mPipeHasAnimation || mActivePasses[passNum]->hasAnimation();
        }
    }

    mPipeRequiresScene = mPipeRequiresScene || mPipeNeedsDefaultScene;
}

void RenderingPipeline::createDefaultDropdownGuiForPass(uint32_t passOrder, Gui::DropdownList& outputList)
{
    // Ensure our list is empty
    outputList.resize(0);

    // Add an item to allow selecting a null pass.
    outputList.push_back({ kNullPassId, kNullPassDescriptor });

    // Include an item in the dropdown for each possible pass
    for (uint32_t i = 0; i < mAvailPasses.size(); i++)
    {
        // Don't add if this pass doesn't exist or if we can't insert it at that location in the pipeline.
        if (!mAvailPasses[i]) continue;
        if (!isPassValid(mAvailPasses[i], passOrder)) continue;

        // Ok.  The pass exists and is able to be selected for pass number <passOrder>.  Insert it into the list.
        outputList.push_back({ i, mAvailPasses[i]->getName() });
    }
}

// Returns true if pCheckPass is valid to insert in the pass sequence at location <passNum>
bool RenderingPipeline::isPassValid(::RenderPass::SharedPtr pCheckPass, uint32_t passNum)
{
    // For now, say that all passes can be inserted everywhere...
    return true;
}

void RenderingPipeline::onGuiRender(Gui* pGui)
{
    //Falcor::ProfilerEvent _profileEvent("renderGUI");

    Gui::Window w(pGui, "Falcor", { 300, 800 }, { 10, 80 });
    gpFramework->renderGlobalUI(pGui);

    w.separator();

    // Add a button to allow the user to load a scene
    if (mPipeRequiresScene)
    {
        w.text("Need to open a new scene?  Click below:");
        w.text("     ");
        if (w.button("Load Scene", true))
        {
            // A wrapper function to open a window, load a UI, and do some sanity checking
            Scene::SharedPtr loadedScene = loadScene(mLastKnownSize);

            // We have a method that explicitly initializes all render passes given our new scene.
            if (loadedScene)
            {
                onInitNewScene(gpFramework->getRenderContext(), loadedScene);
                mGlobalPipeRefresh = true;
            }
        }
        w.separator();
    }

    if (mPipeUsesEnvMap)
    {
        // Just in case we got here without having populated the selector
        if (mEnvMapSelector.empty()) populateEnvMapSelector();

        uint32_t selection = 0;
        w.text( "Current environment map:" );
        w.text("     ");
        if (w.dropdown("##envMapSelector", mEnvMapSelector, selection, true))
        {
            if (selection == 1)  // Then we've asked to load a new map
            {
                bool isValid = false;
                std::string fileName = getTextureLocation(isValid);
                if (isValid && mpResourceManager->updateEnvironmentMap(fileName))
                {
                    mEnvMapSelector[0] = { 0, mpResourceManager->getEnvironmentMapName().c_str() };
                }
            }
            else if (selection == 2)  // select default "black" environment
            {
                // Choose the black background
                mpResourceManager->updateEnvironmentMap("Black");
                mEnvMapSelector[0] = { 0, "Black (i.e., [0.0, 0.0, 0.0])" };
            }
            else if (selection == 3)  // select default "sky blue" environment
            {
                mpResourceManager->updateEnvironmentMap("");
                mEnvMapSelector[0] = { 0, "Sky blue (i.e., [0.5, 0.5, 0.8])" };
            }
            else if (selection == 4)
            {
                mEnvMapSelector[0] = { 0, "Desert HDR environment" };
                mpResourceManager->updateEnvironmentMap(mMonValleyFilename);
            }
            mGlobalPipeRefresh = true;
        }
        w.separator();
    }

    if (mPipeRequiresRayTracing && mpResourceManager)
    {
        w.text("Set ray tracing min traversal distance:");
        w.text("     ");
        if (w.dropdown("##minTSelector", mMinTDropdown, mMinTSelection, true))
        {
            mpResourceManager->setMinTDist(mMinTArray[mMinTSelection]);
            mGlobalPipeRefresh = true;
        }
        w.separator();
    }

    // To avoid putting GUIs on top of each other, offset later passes
    int yGuiOffset = 0;

    // Do we have pipeline instructions?
    if (mPipeDescription.size() > 0)
    {
        // Print all the lines in the instructions / help message
        for (auto line : mPipeDescription)
        {
            w.text(line.c_str());
        }

        // Add a blank line.
        w.text(""); 
    }

    // Draw the checkbox that enables/disables all passes' GUIs
    w.text("Ordered list of passes in rendering pipeline:");
    if (w.checkbox("##enableAllGuis", mEnableAllPassGui))
    {
        // This flag will ensure the window position changes are propagated
        mResetWindowPositions = true;
        for (uint32_t i = 0; i < mPassSelectors.size(); i++)
        {
            mEnablePassGui[i] = mEnableAllPassGui;
        }
    }
    w.text(" Display all GUIs", true);

    // Draw pass selectors for each available pass
    for (uint32_t i = 0; i < mPassSelectors.size(); i++)
    {
        char buf[128];

        // Draw a button that enables/disables showing this pass' GUI window
        sprintf_s(buf, "##enable.pass.%d", i);
        bool enableGui = mEnablePassGui[i];
        if (w.checkbox(buf, enableGui)) {
            mEnablePassGui[i] = enableGui;
            // Update the "select all" checkbox as well
            mEnableAllPassGui = allPassGuisEnabled();
        }

        // Draw the selectable list of passes we can add at this stage in the pipeline
        sprintf_s(buf, "##selector.pass.%d", i);
        if (w.dropdown(buf, mPassSelectors[i], mPassId[i], true))
        {
            ::RenderPass::SharedPtr selectedPass = (mPassId[i] != kNullPassId) ? mAvailPasses[mPassId[i]] : nullptr;
            changePass(i, selectedPass);
            // Reset the preset selection, and update the resource manager (in case we were using a preset)
            mSelectedPreset = kNullPresetId;
            mpResourceManager->setCopyOutTextureName("");
        }

        // If the GUI for this pass is enabled, go ahead and draw the GUI
        if (mEnablePassGui[i] && mActivePasses[i])
        {
            // Find out where to put the GUI for this pass
            int2 guiPos = mResetWindowPositions ? int2(-320, 30) : mActivePasses[i]->getGuiPosition();
            int2 guiSz = mActivePasses[i]->getGuiSize();

            // If the GUI position is negative, attach to right/bottom on screen
            guiPos.x = (guiPos.x < 0) ? (mLastKnownSize.x + guiPos.x) : guiPos.x;
            guiPos.y = (guiPos.y < 0) ? (mLastKnownSize.y + guiPos.y) : guiPos.y;

            // Offset the positions of the GUIs depending on where they are in the pipeline.  If we move it down too far
            //    due to number of passes, give up and draw it just barely visibile at the bottom of the screen.
            guiPos.y += yGuiOffset; 
            guiPos.y = glm::min(guiPos.y, int(mLastKnownSize.y) - 100);

            // Create a window.  Note: RS4 version does more; that doesn't work with recent Falcor; this is OK for just tutorials.
            Gui::Window passWindow(pGui, mActivePasses[i]->getGuiName().c_str(), { guiSz.x, guiSz.y }, { guiPos.x, guiPos.y }, mPassWindowFlags);

            // If the flag to reset the window positions was set, we call this to move it to the calculated positon,
            // since the constructor Gui::Window() only sets the position of the window the first time it is called.
            if (mResetWindowPositions)
                passWindow.windowPos(guiPos.x, guiPos.y);

            // Render the pass' GUI to this new UI window, then pop the new UI window.
            mActivePasses[i]->onRenderGui(&passWindow);
            passWindow.release();
        }

        // Offset the next GUI by the current one's height
        if (mActivePasses[i])
            yGuiOffset += mActivePasses[i]->getGuiSize().y; 
    }
    mResetWindowPositions = false;

    // Draw the preset selector
    if (!mPresetSelector.empty())
    {
        w.text("Selected preset sequence:");
        w.text("     ");
        // Draw a button that enables/disables showing this pass' GUI window
        if (w.dropdown("##presetSelector", mPresetSelector, mSelectedPreset, true))
        {
            // If a preset is selected, reset all the window positions
            mResetWindowPositions = true;

            // Reset the texture reference in the resource manager that indicates to the CopyToOutputPass which texture to copy
            mpResourceManager->setCopyOutTextureName("");

            // Ignore the null selection 
            if (mSelectedPreset != kNullPresetId)
            {
                selectPreset();
            }
        }
    }
    w.text("");

    w.separator();
    w.text(Falcor::gProfileEnabled ? "Press (P):  Hide profiling window" : "Press (P):  Show profiling window");
    w.separator();

    if (mpScene) mpScene->renderUI(w);
}

void RenderingPipeline::selectPreset()
{
    const std::vector<uint32_t>& selectedPassIdxs = mPresetsData[mSelectedPreset].selectedPassIdxs;
    for (uint32_t i = 0; i < selectedPassIdxs.size(); i++)
    {
        // The ith element is the index of the pass selector to choose
        uint32_t idx = selectedPassIdxs[i];
        // Get the underlying pass index from the pass selector.
        uint32_t passIdx = mPassSelectors[i][idx].value;
        mPassId[i] = passIdx;

        ::RenderPass::SharedPtr selectedPass = (mPassId[i] != kNullPassId) ? mAvailPasses[mPassId[i]] : nullptr;
        changePass(i, selectedPass);
    }
    mpResourceManager->setCopyOutTextureName(mPresetsData[mSelectedPreset].outBuf);
    mGlobalPipeRefresh = true;
}

void RenderingPipeline::removePassFromPipeline(uint32_t passNum)
{
    // Check index validity (and don't allow removal of the last list entry)
    if (passNum >= mActivePasses.size() - 1)
    {
        return;
    }

    // If we're removing an active pass, deactive the current one.
    if (mActivePasses[passNum])
    {
        // Tell the pass that it's been deactivated
        mActivePasses[passNum]->onPassDeactivation();
    }

    // Remove entry from all the internal lists
    mActivePasses.erase(mActivePasses.begin() + passNum);
    mPassSelectors.erase(mPassSelectors.begin() + passNum);
    mPassId.erase(mPassId.begin() + passNum);
    mEnablePassGui.erase(mEnablePassGui.begin() + passNum);
    mEnableAddRemove.erase(mEnableAddRemove.begin() + passNum);

    // (Re)-create a GUI selector for all passes (after the removed one)  
    for (uint32_t i = 0; i < mPassSelectors.size(); i++)
    {
        if (mPassSelectors[i].size() <= 0)
            createDefaultDropdownGuiForPass(i, mPassSelectors[i]);
    }

    // The pipeline has changed, so make sure to let people know.
    mPipelineChanged = true;
}

void RenderingPipeline::setPass(uint32_t passNum, ::RenderPass::SharedPtr pTargetPass, bool canAddPassAfter, bool canRemovePass)
{
    // If we're setting pass after last pass in the list, insert null passes so that all slots are valid.
    for (uint32_t i = (uint32_t)mPassId.size(); i <= passNum; i++)
    {
        insertPassIntoPipeline(i);
    }

    // Get unique pass index. Add pass to list of available passes.
    uint32_t passIdx = kNullPassId;
    if (pTargetPass)
    {
        // Find if this pass is in our existing list of available passes.
        auto passLoc = std::find(mAvailPasses.begin(), mAvailPasses.end(), pTargetPass);
        passIdx = uint32_t(passLoc - mAvailPasses.begin());

        // If it is not in the list of available passes, then add it.
        if (passLoc == mAvailPasses.end())
        {
            passIdx = addPass(pTargetPass);
        }
    }

    // Create a GUI dropdown for this new pass
    mPassSelectors[passNum].resize(0);
    mPassSelectors[passNum].push_back({ kNullPassId, kNullPassDescriptor });
    if (pTargetPass && passIdx != kNullPassId)
    {
        mPassSelectors[passNum].push_back({ passIdx, mAvailPasses[passIdx]->getName() });
    }

    // Update the settings for the specified active pass.
    mPassId[passNum] = passIdx;
    mEnableAddRemove[passNum] = (canAddPassAfter ? UIOptions::CanAddAfter : 0x0u) | (canRemovePass ? UIOptions::CanRemove : 0x0u);

    if (mIsInitialized)
    {
        // If setPass() was called after initialization, call changePass() to properly active/deactive passes.
        changePass(passNum, pTargetPass);
    }
    else
    {
        // If not initialized, just store the pass ptr. It will be resized/actived later.
        // We do this way so that callbacks are not called on the pass before the device is created.
        // The initialize() function should always be the first to be called.
        mActivePasses[passNum] = pTargetPass;
    }

    // The pipeline has changed, so set the flag.
    mPipelineChanged = true;
}

void RenderingPipeline::setPassOptions(uint32_t passNum, std::vector<::RenderPass::SharedPtr> pPassList)
{
    // If we're setting pass after last pass in the list, insert null passes so that all slots are valid.
    for (uint32_t i = (uint32_t)mPassId.size(); i <= passNum; i++)
    {
        insertPassIntoPipeline(i);
    }

    // Can't do anything else if the list of passes to choose from for this pass is null-length
    if (pPassList.size() <= 0) return;

    // Create a GUI dropdown for this new pass
    mPassSelectors[passNum].resize(0);
    mPassSelectors[passNum].push_back({ kNullPassId, kNullPassDescriptor });

    // Update the settings for the specified active pass.
    mEnableAddRemove[passNum] = 0x0u; 

    uint32_t passIdx = kNullPassId;
    for (uint32_t i = 0; i < uint32_t(pPassList.size()); i++)
    {
        // Get unique pass index. Add pass to list of available passes.
        if (pPassList[i])
        {
            // Find if this pass is in our existing list of available passes.
            auto passLoc = std::find(mAvailPasses.begin(), mAvailPasses.end(), pPassList[i]);
            passIdx = uint32_t(passLoc - mAvailPasses.begin());

            // If it is not in the list of available passes, then add it.
            if (passLoc == mAvailPasses.end())
            {
                passIdx = addPass(pPassList[i]);
            }

            if (passIdx != kNullPassId)
            {
                mPassSelectors[passNum].push_back({ passIdx, mAvailPasses[passIdx]->getName() });
            }
        }

        // Set the active pass to be the first one in the list
        if (i==0) mPassId[passNum] = passIdx;
    }

    if (mIsInitialized)
    {
        // If setPass() was called after initialization, call changePass() to properly active/deactive passes.
        changePass(passNum, pPassList[0]);
    }
    else
    {
        // If not initialized, just store the pass ptr. It will be resized/actived later.
        // We do this way so that callbacks are not called on the pass before the device is created.
        // The initialize() function should always be the first to be called.
        mActivePasses[passNum] = pPassList[0];
    }

    // The pipeline has changed, so set the flag.
    mPipelineChanged = true;
}

void RenderingPipeline::insertPassIntoPipeline(uint32_t afterPass)
{
    // Since std::vector::insert() inserts *before* the specified element, we need to use afterPass+1.  
    //    But if inserting into an empty list (or potentially at the end of a list), this causes problems,
    //    so do some index clamping.
    uint32_t insertLoc = (afterPass < mActivePasses.size()) ? afterPass + 1 : uint32_t(mActivePasses.size());

    // Insert new null entry into the list
    Gui::DropdownList nullSelector;
    mActivePasses.insert(mActivePasses.begin() + insertLoc, nullptr);
    mPassSelectors.insert(mPassSelectors.begin() + insertLoc, nullSelector);
    mPassId.insert(mPassId.begin() + insertLoc, kNullPassId);
    mEnablePassGui.insert(mEnablePassGui.begin() + insertLoc, false);
    mEnableAddRemove.insert(mEnableAddRemove.begin() + insertLoc, int32_t( UIOptions::CanAddAfter | UIOptions::CanRemove ) );

    // (Re)-create a GUI selector for all passes (including and after the new pass)  
    for (uint32_t i = insertLoc; i < mPassSelectors.size(); i++)
    {
        if (mPassSelectors[i].size() <= 0)
            createDefaultDropdownGuiForPass(i, mPassSelectors[i]);
    }
}

void RenderingPipeline::changePass(uint32_t passNum, ::RenderPass::SharedPtr pNewPass)
{
    // Early out if the new pass is the same as the old pass.
    if (mActivePasses[passNum] && mActivePasses[passNum] == pNewPass)
    {
        return;
    }

    // If we're changing an active pass to a different one, deactivate the current pass
    if (mActivePasses[passNum])
    {
        // Tell the pass that it's been deactivated
        mActivePasses[passNum]->onPassDeactivation();
    }

    // Set the selected active pass
    mActivePasses[passNum] = pNewPass;

    // Do any activation of the newly selected pass (if it's non-null)
    if (pNewPass)
    {
        // Ensure the pass knows the correct size
        pNewPass->onResize(mLastKnownSize.x, mLastKnownSize.y);

        // Tell the pass that it's been activated
        pNewPass->onPassActivation();
    }

    // (Re)-create a GUI selector for all passes (including any newly added one)
    //    We recreate all selectors in case the pass change we're now processing
    //    affects the validity of selections in previously-created lists.
    for (uint32_t i = 0; i < mPassSelectors.size(); i++)
    {
        if (mPassSelectors[i].size() <= 0)
            createDefaultDropdownGuiForPass(i, mPassSelectors[i]);
    }

    // If we've changed a pass, the pipeline has changed
    updatePipelineRequirementFlags();
    mPipelineChanged = true;
}

void RenderingPipeline::onFirstRun()
{
    // Did the user ask for us to load a scene by default?
    if (mPipeNeedsDefaultScene)
    {
        Scene::SharedPtr loadedScene = loadScene(mLastKnownSize, mpResourceManager->getDefaultSceneName().c_str());
        if (loadedScene) onInitNewScene(gpFramework->getRenderContext(), loadedScene);
    }

    // By default, select the first preset if it exists
    if (!mPresetSelector.empty())
    {
        mSelectedPreset = 0;
        selectPreset();
    }

    mFirstFrame = false;
}

void RenderingPipeline::onFrameRender(RenderContext* pRenderContext, const std::shared_ptr<Fbo>& pTargetFbo)
{
    // Is this the first time we've run onFrameRender()?  If som take care of things that happen on first execution.
    if (mFirstFrame) onFirstRun();
    
    // Check to ensure we have all our resources initialized.  (This should be superfluous) 
    if (!mpResourceManager->isInitialized())
    {
        mpResourceManager->initializeResources();
    }

    // If we have a scene, make sure to update the current camera based on any UI controls
    if (mpScene)
    {
        // Update the scene
        mpScene->update(pRenderContext, gpFramework->getGlobalClock().getTime());
    }

    // Check if the pipeline has changed since last frame and needs updating
    bool updatedPipeline = false;
    if (anyRequestedPipelineChanges())
    {
        // If there's a change, let all the passes know
        for (uint32_t passNum = 0; passNum < mActivePasses.size(); passNum++)
        {
            if (mActivePasses[passNum])
            {
                mActivePasses[passNum]->onPipelineUpdate( mpResourceManager );
            }
        }

        // Update our flags
        updatePipelineRequirementFlags();
        updatedPipeline = true;
    }

    // Check if any passes have set their refresh flag
    if (havePassesSetRefreshFlag() || updatedPipeline || mGlobalPipeRefresh)
    {
        // If there's a change, let all the passes know
        for (uint32_t passNum = 0; passNum < mActivePasses.size(); passNum++)
        {
            if (mActivePasses[passNum])
            {
                mActivePasses[passNum]->onStateRefresh();
            }
        }
        mGlobalPipeRefresh = false;
    }

    // Execute all of the passes in the current pipeline
    for (uint32_t passNum = 0; passNum < mActivePasses.size(); passNum++)
    {
        if (mActivePasses[passNum])
        {
            if (Falcor::gProfileEnabled)
            {
                // Insert a per-pass profiling event.  
                assert(passNum < mProfileNames.size());
                Falcor::ProfilerEvent _profileEvent(mActivePasses[passNum]->getName().c_str());
                mActivePasses[passNum]->onExecute(pRenderContext);
            }
            else
            {
                mActivePasses[passNum]->onExecute(pRenderContext);
            }
        }
    }

    // Now that we're done rendering, grab out output texture and blit it into our target FBO
    if (pTargetFbo && mpResourceManager->getTexture(mOutputBufferIndex))
    {
        pRenderContext->blit(mpResourceManager->getTexture(mOutputBufferIndex)->getSRV(), pTargetFbo->getColorTexture(0)->getRTV());
    }

    // Once we're done rendering, clear the pipeline dirty state.
    mPipelineChanged = false;

    // Print the FPS
    TextRenderer::render(pRenderContext, gpFramework->getFrameRate().getMsg(), pTargetFbo, { 20, 20 });
}

void RenderingPipeline::onInitNewScene(RenderContext* pRenderContext, Scene::SharedPtr pScene)
{
    if (pScene) {
        // Stash the scene in the pipeline
        mpScene = pScene;
        // Create a camera controller
        mpScene->setCameraController(Scene::CameraControllerType::FirstPerson);
    }

    // When a new scene is loaded, we'll tell all our passes about it (not just active passes)
    for (uint32_t i = 0; i < mAvailPasses.size(); i++)
    {
        if (mAvailPasses[i])
        {
            mAvailPasses[i]->onInitScene(pRenderContext, pScene);
        }
    }
}

void RenderingPipeline::onResizeSwapChain(uint32_t width, uint32_t height)
{
    // Stash the current size, so if we need it later, we'll have access.
    mLastKnownSize = uint2(width, height);

    // If we're getting zeros for width or height, we're don't have a real screen yet and we're
    //    going to get lots of resource resizing issues.  Stop until we have a reasonable size
    if (width <= 0 || height <= 0) return;

    // Resizes our resource manager
    if (mpResourceManager)
    {
        mpResourceManager->resize(width, height);
    }

    // We're only going to resize render passes that are active.  Other passes will get resized when activated.
    for (uint32_t i = 0; i < mActivePasses.size(); i++)
    {
        if (mActivePasses[i])
        {
            mActivePasses[i]->onResize(width, height);
        }
    }
}

void RenderingPipeline::onShutdown()
{
    // On program shutdown, call the shutdown callback on all the render passes.
    // We do not have to worry about double-deletion etc. It is currently enforced that a pass is only bound to one pipeline.
    for (uint32_t i = 0; i < mAvailPasses.size(); i++)
    {
        if (mAvailPasses[i])
        {
            mAvailPasses[i]->onShutdown();
        }
    }
}

bool RenderingPipeline::onKeyEvent(const KeyboardEvent& keyEvent)
{
    // Let all of the currently active render passes process any keyboard events
    for (uint32_t i = 0; i < mActivePasses.size(); i++)
    {
        if (mActivePasses[i] && mActivePasses[i]->onKeyEvent(keyEvent))
        {
            return true;
        }
    }
    return mpScene ? mpScene->onKeyEvent(keyEvent) : false;
}

bool RenderingPipeline::onMouseEvent(const MouseEvent& mouseEvent)
{
    // Some odd cases where this gets called by Falcor error message boxes.  Ignore these.
    if (!gpFramework || !mpScene) return false;

    // Let all of the currently active render passes process any mouse events
    for (uint32_t i = 0; i < mActivePasses.size(); i++)
    {
        if (mActivePasses[i] && mActivePasses[i]->onMouseEvent(mouseEvent))
        {
            return true;
        }
    }

    return mpScene->onMouseEvent(mouseEvent);
}

bool RenderingPipeline::canRemovePass(uint32_t passNum)
{
    if (passNum >= mEnableAddRemove.size()) return false;
    return (mEnableAddRemove[passNum] & UIOptions::CanRemove) != 0x0u;
}

bool RenderingPipeline::canAddPassAfter(uint32_t passNum)
{
    if (passNum >= mEnableAddRemove.size()) return false;
    return (mEnableAddRemove[passNum] & UIOptions::CanAddAfter) != 0x0u;
}

void RenderingPipeline::getActivePasses(std::vector<::RenderPass::SharedPtr>& activePasses) const
{
    activePasses.clear();
    for (auto& pPass : mActivePasses)
    {
        if (pPass)
        {
            activePasses.push_back(pPass);
        }
    }
}

bool RenderingPipeline::anyRequestedPipelineChanges(void)
{
    // Ask our passes if they've change the pipeline
    for (uint32_t passNum = 0; passNum < mActivePasses.size(); passNum++)
    {
        if (mActivePasses[passNum] && mActivePasses[passNum]->isRebindFlagSet())
        {
            mPipelineChanged = true;
            mActivePasses[passNum]->resetRebindFlag();
        }
    }

    // Ask our resource manager if it has changed any pipeline state
    if (mpResourceManager->haveResourcesChanged())
    {
        mPipelineChanged = true;
        mpResourceManager->resetDirtyFlag();
    }

    return mPipelineChanged;
}

bool RenderingPipeline::havePassesSetRefreshFlag(void)
{
    bool refreshFlag = false;
    for (uint32_t passNum = 0; passNum < mActivePasses.size(); passNum++)
    {
        if (mActivePasses[passNum] && mActivePasses[passNum]->isRefreshFlagSet())
        {
            refreshFlag = true;
        }
    }
    return refreshFlag;
}

bool RenderingPipeline::anyPassUsesEnvMap(void)
{
    for (auto& pass : mAvailPasses)
    {
        if (pass->usesEnvironmentMap())
            return true;
    }
    return false;
}

bool RenderingPipeline::allPassGuisEnabled(void)
{
    for (uint32_t i = 0; i < mPassSelectors.size(); i++)
        if (!mEnablePassGui[i]) return false;
    return true;
}

void RenderingPipeline::populateEnvMapSelector(void)
{
    // Check to see if someone else already loaded one. 
    std::string envName = mpResourceManager->getEnvironmentMapName();
    if (envName == "") // No loaded map?  Create a default one
    {
        mpResourceManager->updateEnvironmentMap("");
        mEnvMapSelector.push_back({ 0, "Sky blue (i.e., [0.5, 0.5, 0.8])" });
    }
    else  // Map already loaded, get it's name to put in the UI
        mEnvMapSelector.push_back({ 0, envName.c_str() });

    // Add a UI option to load a new HDR environment map
    mEnvMapSelector.push_back({ 1, "< Load new map... >" });
    mEnvMapSelector.push_back({ 2, "Switch -> black environment" });
    mEnvMapSelector.push_back({ 3, "Switch -> sky blue environment" });

    if (findFileInDataDirectories("MonValley_G_DirtRoad_3k.hdr", mMonValleyFilename))
    {
        mHasMonValley = true;
        mEnvMapSelector.push_back({ 4, "Switch -> desert HDR environment" });
    }
}

void RenderingPipeline::addPipeInstructions(const std::string &str)
{
    mPipeDescription.push_back(str);
}

void RenderingPipeline::run(RenderingPipeline *pipe, SampleConfig &config)
{
    pipe->updatePipelineRequirementFlags();
    auto uPtrToPipe = std::unique_ptr<IRenderer>(pipe);
    Sample::run(config, uPtrToPipe);
}
