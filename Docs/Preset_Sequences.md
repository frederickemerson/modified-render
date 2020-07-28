### [Index](./index.md) | Getting Started

--------

# Preset Sequences

![Preset sequences](./images/Preset_Sequence.gif)

To create a preset sequence of passes, we need to have already set the passes in the pipeline, otherwise, setting the presets will fail.

For a single preset, we need to create a 1-indexed `vector<uint32_t>`, where each `uint32_t` represents the option to select in the pass.

```
// Assume we have two passes in our pipeline
pipeline->setPassOptions(0, {
        JitteredGBufferPass::create(),
        LightProbeGBufferPass::create()
});

pipeline->setPass(1, GGXGlobalIlluminationPass::create("TextureName"));

// This is our preset options vector
std::vector<uint32_t> preset_options = { 2, 1 };
```

In this example, the preset will select the `LightProbeGBufferPass` for the first pass, and the `GGXGlobalIlluminationPass` (the only option) for the second pass.

Take note that the length of the options vector must be the same as the number of passes in the pipeline, otherwise it will not be added. 

Putting the value `0` in the preset option will choose the "None" option for that pass.

Then, we set the presets in the pipeline:
```
pipeline->setPresets({ 
	RenderingPipeline::PresetData("MyPresetName", "TextureName", preset_options)
);

// Now we can run the program
RenderingPipeline::run(pipeline, config);
```

`setPresets()` takes a vector of `PresetData`. The constructor of `PresetData` takes the option name (displayed in the dropdown menu), a channel name, and the options vector. The channel name can specify the texture we want our viewer to see if we have a `CopyToOutputPass` in the pipeline. When we select the preset, the `CopyToOutputPass` will automatically display the specified texture.

Note that if the presets list is not empty, then on startup, by default the first preset will be displayed.
