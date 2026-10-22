// Final pass: the sharp light layer plus its bloom, with the glitch
// (chromatic aberration, tearing, scanlines) and the hand-off whip blur,
// written as premultiplied light onto the swap chain.

static const int kWhipTaps = 8;
static const float kTearBands = 48.0;
static const float kTearRate = 24.0;
static const float kShoulder = 0.78;

float hash11(float n)
{
    return frac(sin(n * 12.9898) * 43758.5453);
}

// The light layer at uv; while the hand-off whips, taps ahead of the pixel
// along the motion, so every light leaves a trail behind it as it streaks
// into its corner (whip points the way the HUD moves).
float4 lightAt(float2 uv)
{
    if (dot(whip, whip) < 1e-10)
    {
        return source.Sample(linearClamp, uv);
    }
    float4 sum = 0.0;
    float weights = 0.0;
    [unroll] for (int i = 0; i < kWhipTaps; ++i)
    {
        const float t = (float)i / (float)(kWhipTaps - 1);
        const float w = 1.0 - (0.75 * t);
        sum += source.Sample(linearClamp, uv + (whip * t)) * w;
        weights += w;
    }
    return sum / weights;
}

// Soft shoulder: values below kShoulder pass unchanged, brighter ones
// roll off towards 1 instead of clipping to a flat white.
float3 shoulder(float3 c)
{
    const float3 over = max(c - kShoulder, 0.0);
    const float3 rolled = kShoulder + ((1.0 - kShoulder) * (1.0 - exp(-over / (1.0 - kShoulder))));
    return (c < kShoulder) ? c : rolled;
}

float4 main(FullscreenVertex input) : SV_Target
{
    float2 uv = input.uv;
    // Tearing: a few horizontal bands slip sideways while the glitch is high.
    const float band = floor(uv.y * kTearBands);
    const float tick = floor(time * kTearRate);
    const float slip = step(0.7, hash11(band + (tick * 7.13))) * (hash11(band * 1.7 + tick) - 0.5);
    uv.x += slip * glitch * glitch * 0.04;

    const float2 aberration = ((uv - 0.5) * 0.010 + float2(0.003, 0.0)) * glitch;
    const float4 centre = lightAt(uv);
    const float red = lightAt(uv + aberration).r;
    const float blue = lightAt(uv - aberration).b;
    float3 colour = float3(red, centre.g, blue);

    colour += bloomTexture.Sample(linearClamp, input.uv).rgb * bloomStrength * bloomNorm;
    const float scan = 0.5 + (0.5 * sin(input.uv.y * targetHeight * 3.14159265));
    colour *= 1.0 - (glitch * 0.35 * scan);
    colour = shoulder(colour);
    return float4(colour, saturate(maxChannel(colour) * coverage));
}
