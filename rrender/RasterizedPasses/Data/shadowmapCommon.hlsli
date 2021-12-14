#ifndef INTERPOLATION_MODE
#define INTERPOLATION_MODE linear
#endif

struct ShadowMapVertexOut
{
    INTERPOLATION_MODE float3 posW : POSW;
    float4 posH : SV_POSITION;
};

// Based on wikipedia "Cube mapping"
// Takes a direction vector and returns the UV and index of the cube map.
// UV will be in range [0, 1], and must be multiplied by texture dimensions to
// get the pixel coordinates.
// The cubemap is actually a res * (res*6) shadow map.
float3 dirToCubeCoords(float3 dir)
{
    float x = dir.x; float y = dir.y; float z = dir.z;
    int index; float u; float v;
    float absX = abs(x);
    float absY = abs(y);
    float absZ = abs(z);

    int isXPositive = x > 0 ? 1 : 0;
    int isYPositive = y > 0 ? 1 : 0;
    int isZPositive = z > 0 ? 1 : 0;

    float maxAxis, uc, vc;

    // POSITIVE X
    if (isXPositive && absX >= absY && absX >= absZ)
    {
        // u (0 to 1) goes from +z to -z
        // v (0 to 1) goes from -y to +y
        maxAxis = absX;
        uc = -z;
        vc = y;
        index = 0;
    }
    // NEGATIVE X
    if (!isXPositive && absX >= absY && absX >= absZ)
    {
        // u (0 to 1) goes from -z to +z
        // v (0 to 1) goes from -y to +y
        maxAxis = absX;
        uc = z;
        vc = y;
        index = 1;
    }
    // POSITIVE Y
    if (isYPositive && absY >= absX && absY >= absZ)
    {
        // u (0 to 1) goes from -x to +x
        // v (0 to 1) goes from +z to -z
        maxAxis = absY;
        uc = x;
        vc = -z;
        index = 2;
    }
    // NEGATIVE Y
    if (!isYPositive && absY >= absX && absY >= absZ)
    {
        // u (0 to 1) goes from -x to +x
        // v (0 to 1) goes from -z to +z
        maxAxis = absY;
        uc = x;
        vc = z;
        index = 3;
    }
    // POSITIVE Z
    if (isZPositive && absZ >= absX && absZ >= absY)
    {
        // u (0 to 1) goes from -x to +x
        // v (0 to 1) goes from -y to +y
        maxAxis = absZ;
        uc = x;
        vc = y;
        index = 4;
    }
    // NEGATIVE Z
    if (!isZPositive && absZ >= absX && absZ >= absY)
    {
        // u (0 to 1) goes from +x to -x
        // v (0 to 1) goes from -y to +y
        maxAxis = absZ;
        uc = -x;
        vc = y;
        index = 5;
    }

    // Convert range from -1 to 1 to 0 to 1
    u = 0.5f * (uc / maxAxis + 1.0f);
    v = 0.5f * (vc / maxAxis + 1.0f);

    return float3(u, v, index);
}
