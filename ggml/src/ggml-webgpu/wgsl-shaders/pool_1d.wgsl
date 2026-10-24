enable f16;

// GGML_OP_POOL_1D (max or average), matching ggml-cpu/ops.cpp: row ir of src starts at ir * stride_src1 (as on the
// CPU, rows are addressed through dim 1's stride only); one thread per output element. Strides are in elements.

#ifdef SRC_F16
#define SrcType f16
#else
#define SrcType f32
#endif

@group(0) @binding(0) var<storage, read> src: array<SrcType>;
@group(0) @binding(1) var<storage, read_write> dst: array<f32>;

struct Params {
    offset_src: u32,
    offset_dst: u32,
    stride_src1: u32,
    stride_dst1: u32,
    iw: u32,
    ow: u32,
    n_rows: u32,
    k: u32,
    s: u32,
    p: u32,
};
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size(WG_SIZE)
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) num_wg: vec3<u32>) {
    let i = gid.x + num_wg.x * WG_SIZE * gid.y;
    if (i >= params.ow * params.n_rows) {
        return;
    }
    let ow = i % params.ow;
    let ir = i / params.ow;
    let row = params.offset_src + ir * params.stride_src1;
    let base = i32(ow * params.s) - i32(params.p);
#ifdef POOL_MAX
    var res = -3.402823e+38;
#else
    var res = 0.0;
    var count = 0u;
#endif
    for (var ki = 0u; ki < params.k; ki++) {
        let j = base + i32(ki);
        if (j < 0 || j >= i32(params.iw)) {
            continue;
        }
        let v = f32(src[row + u32(j)]);
#ifdef POOL_MAX
        res = max(v, res);
#else
        res += v;
        count++;
#endif
    }
#ifndef POOL_MAX
    res = select(0.0, res / f32(count), count > 0u);
#endif
    dst[params.offset_dst + ow + ir * params.stride_dst1] = res;
}
