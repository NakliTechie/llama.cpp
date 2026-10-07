enable f16;

struct MulMatIdGatherParams {
    offset_ids: u32,

    n_expert: u32,
    n_expert_used: u32,
    n_tokens: u32,

    stride_ids_1: u32,
};

@group(0) @binding(0) var<storage, read_write> ids: array<i32>;        // [n_expert_used, n_tokens]
@group(0) @binding(1) var<storage, read_write> global_gathered_expert_used: array<u32>; // [n_expert][n_tokens]
@group(0) @binding(2) var<storage, read_write> global_gathered_tokens: array<u32>; // [n_expert][n_tokens]
@group(0) @binding(3) var<storage, read_write> gathered_count_ids: array<u32>; // [n_expert]

#ifdef PAGED
// expert -> slot in the paged pool; 0xFFFFFFFF = not resident in this pass
@group(0) @binding(4) var<storage, read_write> slot_map: array<u32>;   // [n_expert]
@group(0) @binding(5) var<uniform> params: MulMatIdGatherParams;
#else
@group(0) @binding(4) var<uniform> params: MulMatIdGatherParams;
#endif

var<workgroup> count:atomic<u32>;

@compute @workgroup_size(WG_SIZE)
fn main(@builtin(workgroup_id) wg_id: vec3<u32>,
        @builtin(local_invocation_id) local_id: vec3<u32>) {

    let thread_id = local_id.x;
    let own_expert = wg_id.x; // the expert assigned to this workgroup
#ifdef PAGED
    // an expert not resident in this pass gathers no tokens, so the main kernel skips it
    let present = slot_map[own_expert] != 0xFFFFFFFFu;
#endif

    if (thread_id == 0u) {
        atomicStore(&count, 0);
    }

    workgroupBarrier();

    for (var i = thread_id;i < params.n_expert_used * params.n_tokens;i += WG_SIZE) {
        let row = i / params.n_expert_used;
        let col = i % params.n_expert_used;
        let expert = u32(ids[params.offset_ids + row * params.stride_ids_1 + col]);
#ifdef PAGED
        if (present && own_expert == expert) {
#else
        if (own_expert == expert) {
#endif
            let pos = atomicAdd(&count, 1u);
            let gathered_id = own_expert * params.n_tokens + pos;
            global_gathered_expert_used[gathered_id] = col;
            global_gathered_tokens[gathered_id] = row;
        }
    }

    workgroupBarrier();

    if (thread_id == 0u) {
        gathered_count_ids[own_expert] = atomicLoad(&count);
    }
}
