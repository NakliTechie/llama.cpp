enable f16;

// GGML_OP_LIGHTNING_INDEXER (DeepSeek-V3.2/V4 sparse attention), matching ggml-cpu/ops.cpp:
//   dst[ik, t, 0, s] = sum_h relu(q[:, h, t, s] . k[:, 0, ik, s]) * w[h, t, s] + mask[ik, t, 0, s % nm3]
// One workgroup per output; its threads split the heads, then reduce. Strides and offsets are in elements.

#ifdef K_F16
#define KType f16
#else
#define KType f32
#endif

@group(0) @binding(0) var<storage, read> q: array<f32>;
@group(0) @binding(1) var<storage, read> k: array<KType>;
@group(0) @binding(2) var<storage, read> w: array<f32>;
@group(0) @binding(3) var<storage, read> mask: array<f16>;
@group(0) @binding(4) var<storage, read_write> dst: array<f32>;

struct Params {
    offset_q: u32,
    offset_k: u32,
    offset_w: u32,
    offset_m: u32,
    offset_dst: u32,
    stride_q1: u32,
    stride_q2: u32,
    stride_q3: u32,
    stride_k2: u32,
    stride_k3: u32,
    stride_w1: u32,
    stride_w3: u32,
    stride_m1: u32,
    stride_m3: u32,
    stride_dst1: u32,
    stride_dst3: u32,
    n_embd: u32,
    n_head: u32,
    n_tokens: u32,
    n_kv: u32,
    n_stream: u32,
    nm3: u32,
};
@group(0) @binding(5) var<uniform> params: Params;

var<workgroup> partial: array<f32, WG_SIZE>;

@compute @workgroup_size(WG_SIZE)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(num_workgroups) num_wg: vec3<u32>,
        @builtin(local_invocation_id) lid: vec3<u32>) {
    let o = wid.x + num_wg.x * wid.y;  // uniform across the workgroup
    let in_range = o < params.n_kv * params.n_tokens * params.n_stream;
    let ik = o % params.n_kv;
    let t  = (o / params.n_kv) % params.n_tokens;
    let s  = o / (params.n_kv * params.n_tokens);

    var score = 0.0;
    if (in_range) {
        let k_base = params.offset_k + ik * params.stride_k2 + s * params.stride_k3;
        for (var h = lid.x; h < params.n_head; h += WG_SIZE) {
            let q_base = params.offset_q + h * params.stride_q1 + t * params.stride_q2 + s * params.stride_q3;
            var qk = 0.0;
            for (var i = 0u; i < params.n_embd; i++) {
                qk += q[q_base + i] * f32(k[k_base + i]);
            }
            score += max(qk, 0.0) * w[params.offset_w + h + t * params.stride_w1 + s * params.stride_w3];
        }
    }
    partial[lid.x] = score;
    workgroupBarrier();
    for (var stride = WG_SIZE / 2u; stride > 0u; stride = stride / 2u) {
        if (lid.x < stride) {
            partial[lid.x] += partial[lid.x + stride];
        }
        workgroupBarrier();
    }
    if (lid.x == 0u && in_range) {
        let m = f32(mask[params.offset_m + ik + t * params.stride_m1 + (s % params.nm3) * params.stride_m3]);
        dst[params.offset_dst + ik + t * params.stride_dst1 + s * params.stride_dst3] = partial[0] + m;
    }
}
