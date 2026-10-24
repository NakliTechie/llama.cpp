enable f16;

// GGML_OP_TOP_K_MASK (DeepSeek-V4 sparse attention), matching ggml-cpu/ops.cpp: each dst row starts all -inf and
// keeps the mask's value at every picked column. A pick is top_k[j] (skipped when out of range or its score is
// -inf); with blocks it expands to the columns blocks[:, pick]; extra[j] adds columns directly. One workgroup per
// row. Only the optional inputs present are bound (HAS_SCORE, HAS_BLOCKS, HAS_EXTRA at BIND_* after dst), because
// one buffer may not be bound twice in a dispatch when a binding is writable. Strides are in elements.

#ifdef MASK_F16
#define MaskType f16
#else
#define MaskType f32
#endif

@group(0) @binding(0) var<storage, read_write> mask: array<MaskType>;
@group(0) @binding(1) var<storage, read_write> top_k: array<i32>;
@group(0) @binding(2) var<storage, read_write> dst: array<MaskType>;
#ifdef HAS_SCORE
@group(0) @binding(BIND_SCORE) var<storage, read_write> score: array<f32>;
#endif
#ifdef HAS_BLOCKS
@group(0) @binding(BIND_BLOCKS) var<storage, read_write> blocks: array<i32>;
#endif
#ifdef HAS_EXTRA
@group(0) @binding(BIND_EXTRA) var<storage, read_write> extra: array<i32>;
#endif

struct Params {
    offset_mask: u32,
    offset_top_k: u32,
    offset_score: u32,
    offset_blocks: u32,
    offset_extra: u32,
    offset_dst: u32,
    stride_mask1: u32,
    stride_mask2: u32,
    stride_mask3: u32,
    mask_ne1: u32,
    mask_ne2: u32,
    mask_ne3: u32,
    stride_dst1: u32,
    stride_dst2: u32,
    stride_dst3: u32,
    stride_top_k0: u32,
    stride_top_k1: u32,
    stride_top_k2: u32,
    stride_top_k3: u32,
    stride_score0: u32,
    stride_score1: u32,
    stride_score2: u32,
    stride_score3: u32,
    stride_blocks0: u32,
    stride_blocks1: u32,
    stride_extra0: u32,
    stride_extra1: u32,
    stride_extra2: u32,
    stride_extra3: u32,
    n_kv: u32,
    ne1: u32,
    ne2: u32,
    n_rows: u32,
    n_top: u32,
    n_blk: u32,
    n_pick: u32,
    n_extra: u32,
};
@group(0) @binding(BIND_PARAMS) var<uniform> params: Params;

fn neg_inf() -> MaskType {
#ifdef MASK_F16
    return bitcast<vec2<f16>>(0xfc00fc00u).x;
#else
    return bitcast<f32>(0xff800000u);
#endif
}

@compute @workgroup_size(WG_SIZE)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(num_workgroups) num_wg: vec3<u32>,
        @builtin(local_invocation_id) lid: vec3<u32>) {
    let ir = wid.x + num_wg.x * wid.y;  // uniform across the workgroup
    if (ir >= params.n_rows) {
        return;
    }
    let i1 = ir % params.ne1;
    let i2 = (ir / params.ne1) % params.ne2;
    let i3 = ir / (params.ne1 * params.ne2);
    let mask_row = params.offset_mask + (i1 % params.mask_ne1) * params.stride_mask1 +
                   (i2 % params.mask_ne2) * params.stride_mask2 + (i3 % params.mask_ne3) * params.stride_mask3;
    let dst_row = params.offset_dst + i1 * params.stride_dst1 + i2 * params.stride_dst2 + i3 * params.stride_dst3;

    for (var c = lid.x; c < params.n_kv; c += WG_SIZE) {
        dst[dst_row + c] = neg_inf();
    }
    storageBarrier();
    workgroupBarrier();

    let top_k_row = params.offset_top_k + i1 * params.stride_top_k1 + i2 * params.stride_top_k2 + i3 * params.stride_top_k3;
    for (var j = lid.x; j < params.n_top; j += WG_SIZE) {
        let p = top_k[top_k_row + j * params.stride_top_k0];
        if (p < 0 || p >= i32(params.n_pick)) {
            continue;
        }
#ifdef HAS_SCORE
        {
            let s = score[params.offset_score + u32(p) * params.stride_score0 + i1 * params.stride_score1 +
                          i2 * params.stride_score2 + i3 * params.stride_score3];
            if (!(s > bitcast<f32>(0xff800000u))) {
                continue;
            }
        }
#endif
#ifndef HAS_BLOCKS
        dst[dst_row + u32(p)] = mask[mask_row + u32(p)];
#else
        for (var k = 0u; k < params.n_blk; k++) {
            let c = blocks[params.offset_blocks + k * params.stride_blocks0 + u32(p) * params.stride_blocks1];
            if (c >= 0 && c < i32(params.n_kv)) {
                dst[dst_row + u32(c)] = mask[mask_row + u32(c)];
            }
        }
#endif
    }

#ifdef HAS_EXTRA
    let extra_row = params.offset_extra + i1 * params.stride_extra1 + i2 * params.stride_extra2 + i3 * params.stride_extra3;
    for (var j = lid.x; j < params.n_extra; j += WG_SIZE) {
        let c = extra[extra_row + j * params.stride_extra0];
        if (c >= 0 && c < i32(params.n_kv)) {
            dst[dst_row + u32(c)] = mask[mask_row + u32(c)];
        }
    }
#endif
}
