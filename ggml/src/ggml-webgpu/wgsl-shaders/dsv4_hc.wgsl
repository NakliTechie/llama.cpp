// DeepSeek-V4 hyper-connection ops, one shader with three entry variants (HC_PRE, HC_POST, HC_COMB), matching
// ggml-cpu/ops.cpp. Strides and offsets are in elements; dim 0 of every tensor is contiguous except where a stride
// for it is passed.

#ifdef HC_PRE
@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read> w: array<f32>;
@group(0) @binding(2) var<storage, read_write> dst: array<f32>;

struct Params {
    offset_x: u32,
    offset_w: u32,
    offset_dst: u32,
    stride_x1: u32,
    stride_x2: u32,
    stride_w0: u32,
    stride_w1: u32,
    stride_w2: u32,
    stride_dst1: u32,
    n_embd: u32,
    hc: u32,
    n_tokens: u32,
    scale: f32,
};
@group(0) @binding(3) var<uniform> params: Params;

// dst[i0, t] = scale * sum_h x[i0, h, t] * weight, weight = w[h, t] or sigmoid(w[i0, h, t]) when GATED
@compute @workgroup_size(WG_SIZE)
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) num_wg: vec3<u32>) {
    let i = gid.x + num_wg.x * WG_SIZE * gid.y;
    if (i >= params.n_embd * params.n_tokens) {
        return;
    }
    let i0 = i % params.n_embd;
    let it = i / params.n_embd;
    var sum = 0.0;
    for (var h = 0u; h < params.hc; h++) {
        let xv = x[params.offset_x + i0 + h * params.stride_x1 + it * params.stride_x2];
#ifdef GATED
        let gv = w[params.offset_w + i0 * params.stride_w0 + h * params.stride_w1 + it * params.stride_w2];
        let wv = 1.0 / (1.0 + exp(-gv));
#else
        let wv = w[params.offset_w + h * params.stride_w0 + it * params.stride_w1];
#endif
        sum += xv * wv;
    }
    dst[params.offset_dst + i0 + it * params.stride_dst1] = params.scale * sum;
}
#endif

#ifdef HC_POST
@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read> residual: array<f32>;
@group(0) @binding(2) var<storage, read> post: array<f32>;
#ifdef HAS_COMB
@group(0) @binding(3) var<storage, read> comb: array<f32>;
@group(0) @binding(4) var<storage, read_write> dst: array<f32>;
#else
@group(0) @binding(3) var<storage, read_write> dst: array<f32>;
#endif

struct Params {
    offset_x: u32,
    offset_r: u32,
    offset_p: u32,
    offset_c: u32,
    offset_dst: u32,
    stride_x1: u32,
    stride_r1: u32,
    stride_r2: u32,
    stride_p0: u32,
    stride_p1: u32,
    stride_c0: u32,
    stride_c1: u32,
    stride_c2: u32,
    stride_dst1: u32,
    stride_dst2: u32,
    n_embd: u32,
    hc: u32,
    n_tokens: u32,
};
#ifdef HAS_COMB
@group(0) @binding(5) var<uniform> params: Params;
#else
@group(0) @binding(4) var<uniform> params: Params;
#endif

// dst[i0, d, t] = x[i0, t] * post[d, t] + sum_s residual[i0, s, t] * comb[d, s, t]   (no comb: + residual[i0, d, t])
@compute @workgroup_size(WG_SIZE)
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) num_wg: vec3<u32>) {
    let i = gid.x + num_wg.x * WG_SIZE * gid.y;
    if (i >= params.n_embd * params.hc * params.n_tokens) {
        return;
    }
    let i0   = i % params.n_embd;
    let idst = (i / params.n_embd) % params.hc;
    let it   = i / (params.n_embd * params.hc);
    var sum = x[params.offset_x + i0 + it * params.stride_x1] * post[params.offset_p + idst * params.stride_p0 + it * params.stride_p1];
#ifdef HAS_COMB
    for (var isrc = 0u; isrc < params.hc; isrc++) {
        let rv = residual[params.offset_r + i0 + isrc * params.stride_r1 + it * params.stride_r2];
        let cv = comb[params.offset_c + idst * params.stride_c0 + isrc * params.stride_c1 + it * params.stride_c2];
        sum += rv * cv;
    }
#else
    sum += residual[params.offset_r + i0 + idst * params.stride_r1 + it * params.stride_r2];
#endif
    dst[params.offset_dst + i0 + idst * params.stride_dst1 + it * params.stride_dst2] = sum;
}
#endif

#ifdef HC_COMB
@group(0) @binding(0) var<storage, read> mixes: array<f32>;
@group(0) @binding(1) var<storage, read> scale: array<f32>;
@group(0) @binding(2) var<storage, read> base: array<f32>;
@group(0) @binding(3) var<storage, read_write> dst: array<f32>;

struct Params {
    offset_m: u32,
    offset_s: u32,
    offset_b: u32,
    offset_dst: u32,
    stride_m0: u32,
    stride_m1: u32,
    stride_s0: u32,
    stride_b0: u32,
    stride_dst0: u32,
    stride_dst1: u32,
    stride_dst2: u32,
    n_tokens: u32,
    n_iter: u32,
    eps: f32,
};
@group(0) @binding(4) var<uniform> params: Params;

const HC: u32 = 4u;
const COMB_OFFSET: u32 = 8u;  // 2*HC

// the 4x4 mixing matrix of one token: softmax over each source column, then Sinkhorn normalisation
@compute @workgroup_size(WG_SIZE)
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) num_wg: vec3<u32>) {
    let it = gid.x + num_wg.x * WG_SIZE * gid.y;
    if (it >= params.n_tokens) {
        return;
    }
    var comb: array<f32, 16>;
    let scale_comb = scale[params.offset_s + 2u * params.stride_s0];
    for (var isrc = 0u; isrc < HC; isrc++) {
        var mx = -3.402823e+38;
        for (var idst = 0u; idst < HC; idst++) {
            let idx = idst + HC * isrc;
            let v = mixes[params.offset_m + (COMB_OFFSET + idx) * params.stride_m0 + it * params.stride_m1] * scale_comb +
                    base[params.offset_b + (COMB_OFFSET + idx) * params.stride_b0];
            comb[idx] = v;
            mx = max(mx, v);
        }
        var sum = 0.0;
        for (var idst = 0u; idst < HC; idst++) {
            let idx = idst + HC * isrc;
            let v = exp(comb[idx] - mx);
            comb[idx] = v;
            sum += v;
        }
        let inv_sum = 1.0 / sum;
        for (var idst = 0u; idst < HC; idst++) {
            let idx = idst + HC * isrc;
            comb[idx] = comb[idx] * inv_sum + params.eps;
        }
    }
    for (var iter = 0u; iter < params.n_iter; iter++) {
        if (iter > 0u) {
            // rows
            for (var isrc = 0u; isrc < HC; isrc++) {
                var sum = params.eps;
                for (var idst = 0u; idst < HC; idst++) {
                    sum += comb[idst + HC * isrc];
                }
                let inv_sum = 1.0 / sum;
                for (var idst = 0u; idst < HC; idst++) {
                    comb[idst + HC * isrc] *= inv_sum;
                }
            }
        }
        // columns
        for (var idst = 0u; idst < HC; idst++) {
            var sum = params.eps;
            for (var isrc = 0u; isrc < HC; isrc++) {
                sum += comb[idst + HC * isrc];
            }
            let inv_sum = 1.0 / sum;
            for (var isrc = 0u; isrc < HC; isrc++) {
                comb[idst + HC * isrc] *= inv_sum;
            }
        }
    }
    for (var isrc = 0u; isrc < HC; isrc++) {
        for (var idst = 0u; idst < HC; idst++) {
            dst[params.offset_dst + idst * params.stride_dst0 + isrc * params.stride_dst1 + it * params.stride_dst2] =
                comb[idst + HC * isrc];
        }
    }
}
#endif
