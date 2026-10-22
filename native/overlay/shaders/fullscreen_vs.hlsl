// One triangle that covers the whole target, generated from SV_VertexID:
// no vertex buffer, no input layout.

FullscreenVertex main(uint id : SV_VertexID)
{
    FullscreenVertex output;
    const float2 uv = float2((id << 1) & 2, id & 2);
    output.position = float4((uv * float2(2.0, -2.0)) + float2(-1.0, 1.0), 0.0, 1.0);
    output.uv = uv;
    return output;
}
