// Bright pass and first downsample (full resolution to half). Four
// bilinear taps average a 4x4 block; each tap is weighted by 1 / (1 + luma)
// (Karis average) so a single very bright pixel cannot flicker the bloom.

float3 softThreshold(float3 colour)
{
    const float brightness = maxChannel(colour);
    float soft = clamp(brightness - threshold + knee, 0.0, 2.0 * knee);
    soft = (soft * soft) / ((4.0 * knee) + 1e-5);
    const float contribution = max(soft, brightness - threshold) / max(brightness, 1e-5);
    return colour * contribution;
}

float4 main(FullscreenVertex input) : SV_Target
{
    const float2 o = sourceTexel;
    const float3 a = source.Sample(linearClamp, input.uv + float2(-o.x, -o.y)).rgb;
    const float3 b = source.Sample(linearClamp, input.uv + float2(o.x, -o.y)).rgb;
    const float3 c = source.Sample(linearClamp, input.uv + float2(-o.x, o.y)).rgb;
    const float3 d = source.Sample(linearClamp, input.uv + float2(o.x, o.y)).rgb;
    const float wa = 1.0 / (1.0 + maxChannel(a));
    const float wb = 1.0 / (1.0 + maxChannel(b));
    const float wc = 1.0 / (1.0 + maxChannel(c));
    const float wd = 1.0 / (1.0 + maxChannel(d));
    const float3 average = ((a * wa) + (b * wb) + (c * wc) + (d * wd)) / (wa + wb + wc + wd);
    return float4(softThreshold(average), 1.0);
}
