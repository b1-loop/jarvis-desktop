// Dual-Kawase upsample: eight taps on a diamond around the pixel, the
// diagonals weighted twice. Blended additively into the next larger mip,
// so every level of the chain contributes its own radius to the glow.

float4 main(FullscreenVertex input) : SV_Target
{
    const float2 o = sourceTexel;
    float3 sum = source.Sample(linearClamp, input.uv + float2(-2.0 * o.x, 0.0)).rgb;
    sum += source.Sample(linearClamp, input.uv + float2(-o.x, o.y)).rgb * 2.0;
    sum += source.Sample(linearClamp, input.uv + float2(0.0, 2.0 * o.y)).rgb;
    sum += source.Sample(linearClamp, input.uv + float2(o.x, o.y)).rgb * 2.0;
    sum += source.Sample(linearClamp, input.uv + float2(2.0 * o.x, 0.0)).rgb;
    sum += source.Sample(linearClamp, input.uv + float2(o.x, -o.y)).rgb * 2.0;
    sum += source.Sample(linearClamp, input.uv + float2(0.0, -2.0 * o.y)).rgb;
    sum += source.Sample(linearClamp, input.uv + float2(-o.x, -o.y)).rgb * 2.0;
    return float4(sum / 12.0, 1.0);
}
