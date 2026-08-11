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
    // TODO: Not yet implemented.
    return;
}

// Used for when there is more than 32 threads per row
kernel void embedding_nsgpr(
    device const float* table    [[buffer(0)]], // [vocab_size, hidden_size]
    device const uint* inVector  [[buffer(1)]], // [batch_size, sequence_length]
    device float* output         [[buffer(2)]], // [batch_size, sequence_length, hidden_size]
    constant uint& hidden_size   [[buffer(3)]],
    constant uint& vocab_size    [[buffer(4)]],
    constant uint& rptg          [[buffer(5)]],
    constant uint& n_tokens      [[buffer(6)]],
    constant uint& sgpr          [[buffer(7)]],
    uint2 tg_idx                 [[thread_position_in_threadgroup]],
    uint2 tg                     [[threadgroup_position_in_grid]],
    uint tpsg                    [[threads_per_simdgroup]],
    uint lane                    [[thread_index_in_simdgroup]]
) {
    // columns per simd group is ceil_div(hidden_size, sgpr)
    uint cpsg = (hidden_size + sgpr - 1) / sgpr;
    // the simd group index (per row) is thread index / simd group size
    uint sg_idx = tg_idx.x / tpsg;
    if (sg_idx > sgpr) return; // Shouldn't happen

    // Compute output row offset
    uint row = tg.y * rptg + tg_idx.y;
    if (row >= n_tokens) return;
    uint output_row_idx = row * hidden_size;
    
    // Compute table row offset
    uint token_id = inVector[row];
    if (token_id >= vocab_size) return; 
    uint table_row_idx = token_id * hidden_size;
    
    // Compute simd group column bounds
    uint sg_start = sg_idx * cpsg;
    uint sg_end = min(sg_start + cpsg, hidden_size);

    // Actual computation
    for (uint h = sg_start + lane; h < sg_end; h += tpsg) {
        output[output_row_idx + h] = table[table_row_idx + h];
    }
}
