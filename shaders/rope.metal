#include <metal_stdlib>
#include <metal_simdgroup>

using namespace metal;

kernel void rope_naive(
    device const float* input       [[buffer(0)]], // [batch, seq_length, num_heads, head_dim]
    device const float* cos_and_sin [[buffer(1)]], // [max_seq_length, rotary_dim / 2, 2 (cos_and_sin)]
    device float* output            [[buffer(2)]], // [batch, seq_length, num_heads, head_dim]
    constant uint& S                [[buffer(3)]], // seq_length
    constant uint& H                [[buffer(4)]], // num_heads
    constant uint& D                [[buffer(5)]], // head_dim
    constant uint& P                [[buffer(6)]], // num_pairs
    constant uint& R                [[buffer(7)]], // rotary_dim = num_pairs * 2
    uint t_idx                      [[thread_position_in_grid]],
    uint grid_size                  [[threads_per_grid]]  // B * S * H
) {
    if (t_idx >= grid_size) return;
    uint seq = (t_idx / H) % S;
    uint offset = t_idx * D;

    for (uint pair = 0; pair < P; ++pair) {
        const uint index = offset + pair;
        const float c = cos_and_sin[seq * P * 2 + pair * 2];
        const float s = cos_and_sin[seq * P * 2 + pair * 2 + 1];
        // Apply rotation (assignmnt to account for in-place substitution)
        float pair_left_output =  input[index] * c - input[index + P] * s;
        float pair_right_output = input[index] * s + input[index + P] * c;
        output[index] = pair_left_output;
        output[index + P] = pair_right_output;
    }

    // Copy the rest
    for (uint d = R; d < D; ++d) { 
        output[offset + d] = input[offset + d];
    }     
}

// Only 1 simd group per row, multiple rows per threadgroup
kernel void rope_1sgpr(
    device const float* input       [[buffer(0)]], // [batch, seq_length, num_heads, head_dim]
    device const float* cos_and_sin [[buffer(1)]], // [max_seq_length, rotary_dim / 2, 2 (cos_and_sin)]
    device float* output            [[buffer(2)]], // [batch, seq_length, num_heads, head_dim]
    constant uint& S                [[buffer(3)]], // seq_length
    constant uint& H                [[buffer(4)]], // num_heads
    constant uint& D                [[buffer(5)]], // head_dim
    constant uint& P                [[buffer(6)]], // num_pairs
    constant uint& R                [[buffer(7)]], // rotary_dim = num_pairs * 2
    uint2 t_idx                     [[thread_position_in_grid]],
    uint2 grid_size                 [[threads_per_grid]],  // B * S * H
    uint tpsg                       [[threads_per_simdgroup]]
) {
    if (t_idx.y >= grid_size.y) return;
    uint seq = (t_idx.y / H) % S;
    uint offset = t_idx.y * D;

    for (uint pair = t_idx.x; pair < P; pair += tpsg) {
        const uint index = offset + pair;
        const float c = cos_and_sin[seq * P * 2 + pair * 2];
        const float s = cos_and_sin[seq * P * 2 + pair * 2 + 1];
        // Apply rotation (assignmnt to account for in-place substitution)
        float pair_left_output =  input[index] * c - input[index + P] * s;
        float pair_right_output = input[index] * s + input[index + P] * c;
        output[index] = pair_left_output;
        output[index + P] = pair_right_output;
    }

    // Copy the rest
    for (uint d = R + t_idx.x; d < D; d += tpsg) { 
        output[offset + d] = input[offset + d];
    }     
}

// N simd groups per row, one row per threadgroup
kernel void rope_nsgpr(
    device const float* input       [[buffer(0)]], // [batch, seq_length, num_heads, head_dim]
    device const float* cos_and_sin [[buffer(1)]], // [max_seq_length, rotary_dim / 2, 2 (cos_and_sin)]
    device float* output            [[buffer(2)]], // [batch, seq_length, num_heads, head_dim]
    constant uint& S                [[buffer(3)]], // seq_length
    constant uint& H                [[buffer(4)]], // num_heads
    constant uint& D                [[buffer(5)]], // head_dim
    constant uint& P                [[buffer(6)]], // num_pairs
    constant uint& R                [[buffer(7)]], // rotary_dim = num_pairs * 2
    constant uint& sgpr             [[buffer(8)]],
    uint2 t_idx                     [[thread_position_in_grid]], // index.x: lane, index.y: row
    uint2 grid_size                 [[threads_per_grid]],
    uint tpsg                       [[threads_per_simdgroup]],
    uint lane                       [[thread_index_in_simdgroup]]
) {
    if (t_idx.y >= grid_size.y) return;
    // columns per simd group for rotary dim is ceil_div(num_pairs, sgpr)
    uint cpsg = (P + sgpr - 1) / sgpr;
    // the simd group index (per row) is thread index / simd group size
    uint sg_idx = t_idx.x / tpsg;
    uint sg_start = sg_idx * cpsg;
    uint sg_end = min(sg_start + cpsg, P);

    uint seq = (t_idx.y / H) % S;
    uint row_idx = t_idx.y * D;

    for (uint pair = sg_start + lane; pair < sg_end; pair += tpsg) {
        const uint index = row_idx + pair;
        const float c = cos_and_sin[seq * P * 2 + pair * 2];
        const float s = cos_and_sin[seq * P * 2 + pair * 2 + 1];
        // Apply rotation (assignmnt to account for in-place substitution)
        float pair_left_output =  input[index] * c - input[index + P] * s;
        float pair_right_output = input[index] * s + input[index + P] * c;
        output[index] = pair_left_output;
        output[index + P] = pair_right_output;
    }

    // Copy the rest
    uint remaining_dim = D - R;
    cpsg = (remaining_dim + sgpr - 1) / sgpr;
    sg_start = sg_idx * cpsg;
    sg_end = min(sg_start + cpsg, remaining_dim);

    for (uint d = R + sg_start + lane; d < R + sg_end; d += tpsg) { 
        output[row_idx + d] = input[row_idx + d];
    } 
}
