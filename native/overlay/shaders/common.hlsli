// Shared by every post-processing pass of the overlay (prepended at build
// time by native/overlay/CMakeLists.txt, so D3DCompile needs no include
// handler). Shader model 4.0 so feature level 10_0 and WARP run it too.

cbuffer PostParams : register(b0)
{
    float2 sourceTexel;   // 1 / size of the texture being read
    float threshold;      // bright-pass threshold (premultiplied light)
    float knee;           // soft knee width around the threshold
    float bloomStrength;  // how much of the blurred light is added back
    float glitch;         // 0 most of the time, spikes on boot beats and the wake pulse
    float time;           // seconds, for the glitch's tear pattern
    float targetHeight;   // output height in pixels, for the scanlines
    float2 whip;          // motion-blur offset in UV for the hand-off whip pan
    float coverage;       // alpha = brightest channel * coverage
    float bloomNorm;      // 1 / number of mip levels added up
};

Texture2D source : register(t0);
Texture2D bloomTexture : register(t1);
SamplerState linearClamp : register(s0);

struct FullscreenVertex
{
    float4 position : SV_Position;
    float2 uv : TEXCOORD0;
};

float maxChannel(float3 c)
{
    return max(c.r, max(c.g, c.b));
}
