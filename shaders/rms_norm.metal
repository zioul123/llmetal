#include <metal_stdlib>
#include <metal_simdgroup>

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

// Only 1 simd group per row, multiple rows per threadgroup
kernel void rms_norm_1sgpr(
    device const float* input    [[buffer(0)]], // [batch_size, sequence_length, hidden_size]
    device const float* weights  [[buffer(1)]], // [hidden_size]
    device float* output         [[buffer(2)]], // [batch_size, sequence_length, hidden_size]
    constant uint& hidden_size   [[buffer(3)]],
    constant float& epsilon      [[buffer(4)]],
    uint2 index                  [[thread_position_in_grid]], // index.x: lane, index.y: row
    uint2 grid_size              [[threads_per_grid]],
    uint tpsg                    [[threads_per_simdgroup]]
) {
    if (index.y >= grid_size.y) return;

    uint offset_start = index.y * hidden_size;
    
    float sum = 0.0f;
    for (uint h = index.x; h < hidden_size; h += tpsg) {
        float curr = input[offset_start + h];
        sum += curr * curr;
    }
    sum = simd_sum(sum);
    float inv_rms = rsqrt(sum / hidden_size + epsilon);

    for (uint h = index.x; h < hidden_size; h += tpsg) {
        output[offset_start + h] = input[offset_start + h] * inv_rms * weights[h];
    }
}
