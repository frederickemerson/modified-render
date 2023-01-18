#pragma once
#include "RenderConfig.h"

RenderConfiguration getDebugRenderConfig(RenderMode mode, RenderType type, unsigned char sceneIdx);
RenderConfiguration getServerRenderConfig(RenderMode mode, RenderType type, unsigned char sceneIdx);
RenderConfiguration getClientRenderConfig(RenderMode mode, RenderType type, unsigned char sceneIdx);