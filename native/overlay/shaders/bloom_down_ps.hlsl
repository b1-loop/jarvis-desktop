// Dual-Kawase downsample (Bjorge, SIGGRAPH 2015): the centre and four
// diagonal bilinear taps, half a target texel out. Five fetches give a
// wide, smooth kernel at half the cost of a separable Gaussian.

float4 main(FullscreenVertex input) : SV_Target
{
    const float2 o = sourceTexel;
    float3 sum = source.Sample(linearClamp, input.uv).rgb * 4.0;
    sum += source.Sample(linearClamp, input.uv - o).rgb;
    sum += source.Sample(linearClamp, input.uv + o).rgb;
    sum += source.Sample(linearClamp, input.uv + float2(o.x, -o.y)).rgb;
    sum += source.Sample(linearClamp, input.uv - float2(o.x, -o.y)).rgb;
    return float4(sum / 8.0, 1.0);
}
