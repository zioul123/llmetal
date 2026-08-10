#include <metal_stdlib>

using namespace metal;

kernel void rms_norm_naive(
    device const float* input    [[buffer(0)]], // [batch_size, sequence_length, hidden_size]
    device const float* weights  [[buffer(1)]], // [hidden_size]
    device float* output         [[buffer(2)]], // [batch_size, sequence_length, hidden_size]
    constant uint& hidden_size   [[buffer(3)]],
    constant float& epsilon      [[buffer(4)]],
    uint index                   [[thread_position_in_grid]],
    uint grid_size               [[threads_per_grid]]
) {
    if (index >= grid_size) return;

    uint offset = index * hidden_size;
    
    float sum = 0.0f;
    for (uint h = 0; h < hidden_size; ++h) {
        float curr = input[offset + h];
        sum += curr * curr;
    }
    float rms = sqrt(sum / hidden_size + epsilon);
    for (uint h = 0; h < hidden_size; ++h) {
        output[offset + h] = input[offset + h] / rms * weights[h];
    }
        
}
