#include <metal_stdlib>
#include <metal_simdgroup>

kernel void rope_naive(
    device const float* input       [[buffer(0)]], // [batch, seq_length, num_heads, head_dim]
    device const float* cos_and_sin [[buffer(1)]], // [max_seq_length, rotary_dim / 2, 2 (cos_and_sin)]
    device float* output            [[buffer(2)]], // [batch, seq_length, num_heads, head_dim]
    constant uint& B                [[buffer(3)]], // batch_size
    constant uint& H                [[buffer(4)]], // num_heads
    constant uint& D                [[buffer(5)]], // head_dim
    constant uint& P                [[buffer(6)]], // num_pairs
    constant uint& R                [[buffer(7)]], // rotary_dim = num_pairs * 2
    uint t_idx                      [[thread_position_in_grid]], // H * S * B
    uint grid_size                  [[threads_per_grid]]
) {
    if (t_idx >= grid_size) return;
    uint seq = t_idx / (H * B);
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