#include <metal_stdlib>
#include <metal_simdgroup>

using namespace metal;

kernel void softmax_naive(
    device const float* logits[[buffer(0)]], // [rows, sequence_length]
    device const uint* valid  [[buffer(1)]], // [rows]
    device float* probs       [[buffer(2)]], // [rows, sequence_length]
    constant uint& seq_length [[buffer(3)]],
    uint index                [[thread_position_in_grid]],
    uint grid_size            [[threads_per_grid]]
) {
    if (index >= grid_size) return;

    uint row_offset = index * seq_length;
    uint valid_seq = valid[index];
    
    // Get the max
    float r_max = -INFINITY;
    for (uint s = 0; s < valid_seq; ++s) {
        r_max = max(r_max, logits[row_offset + s]);
    }

    // Exponentiate
    float factor = 0.0f;
    for (uint s = 0; s < valid_seq; ++s) {
        uint r_c = row_offset + s;
        float curr_val = exp(logits[r_c] - r_max);
        probs[row_offset + s] = curr_val;
        factor += curr_val;
    }

    // Normalize
    factor = 1.0f / factor; // Invert for multiplication
    for (uint s = 0; s < valid_seq; ++s) {
        probs[row_offset + s] *= factor;
    }        
}