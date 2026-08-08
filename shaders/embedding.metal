#include <metal_stdlib>

using namespace metal;

kernel void embedding_naive(
    device const float* table    [[buffer(0)]], // [vocab_size, hidden_size]
    device const uint* inVector  [[buffer(1)]], // [batch_size, sequence_length]
    device float* output         [[buffer(2)]], // [batch_size, sequence_length, hidden_size]
    constant uint& hidden_size   [[buffer(3)]],
    constant uint& vocab_size    [[buffer(4)]],
    uint index                   [[thread_position_in_grid]],
    uint grid_size               [[threads_per_grid]]
) {
    if (index >= grid_size) return;

    uint token_id = inVector[index];
    if (token_id >= vocab_size) return; 

    uint input_offset = token_id * hidden_size;
    uint output_offset = index * hidden_size;
    for (uint h = 0; h < hidden_size; ++h) {
        output[output_offset + h] = table[input_offset + h];
    }
}

// Used for when there is less than 32 threads per thread group
kernel void embedding_tpr(
    device const float* table    [[buffer(0)]], // [vocab_size, hidden_size]
    device const uint* inVector  [[buffer(1)]], // [batch_size, sequence_length]
    device float* output         [[buffer(2)]], // [batch_size, sequence_length, hidden_size]
    constant uint& hidden_size   [[buffer(3)]],
    constant uint& vocab_size    [[buffer(4)]],
    uint index                   [[thread_position_in_grid]],
    uint grid_size               [[threads_per_grid]]
) {
    if (index >= grid_size) return;

    uint token_id = inVector[index];
    if (token_id >= vocab_size) return; 
    
    uint input_offset = token_id * hidden_size;
    uint output_offset = index * hidden_size;
    for (uint h = 0; h < hidden_size; ++h) {
        output[output_offset + h] = table[input_offset + h];
    }
}

// Used for when there is more than 32 threads per row
kernel void embedding_nsgpr(
    device const float* table    [[buffer(0)]], // [vocab_size, hidden_size]
    device const uint* inVector  [[buffer(1)]], // [batch_size, sequence_length]
    device float* output         [[buffer(2)]], // [batch_size, sequence_length, hidden_size]
    constant uint& hidden_size   [[buffer(3)]],
    constant uint& vocab_size    [[buffer(4)]],
    constant uint& rptg          [[buffer(5)]],
    constant uint& tpr           [[buffer(6)]],
    constant uint& n_tokens      [[buffer(7)]],
    constant uint& sgpr          [[buffer(8)]],
    uint2 tgpg                   [[threadgroups_per_grid]],
    uint2 tg_idx                 [[thread_position_in_threadgroup]],
    uint2 tg                     [[threadgroup_position_in_grid]],
    uint2 tptg                   [[threads_per_threadgroup]],
    uint tpsg                    [[threads_per_simdgroup]],
    uint lane                    [[thread_index_in_simdgroup]]
) {
    // columns per simd group is ceil_div(hidden_size, sgpr)
    uint cpsg = (hidden_size + sgpr - 1) / sgpr;
    // the simd group index (per row) is thread index / simd group size
    uint sg_idx = tg_idx.x / tpsg;

    uint row = tg.y * rptg + tg_idx.y;
    if (row >= n_tokens) return;
    
    uint token_id = inVector[row];
    if (token_id >= vocab_size) return; 
    
    // uint input_offset = token_id * hidden_size + sg_idx * cpsg + lane;
    // uint output_offset = row * hidden_size + sg_idx * cpsg + lane;
    
    // for (uint h = 0; h < cpsg && h < ; h += tpsg) {
    //     output[output_offset + h] = table[input_offset + h];
    // }

    uint input_offset = token_id * hidden_size + sg_idx * cpsg;
    uint output_offset = row * hidden_size + sg_idx * cpsg;
    for (uint h = lane; h < cpsg; h += tpsg) {
        output[output_offset + h] = table[input_offset + h];
    }
}
