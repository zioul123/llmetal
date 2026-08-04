#include <metal_stdlib>

using namespace metal;

kernel void gemv_naive(
    device const float* inMatrix [[buffer(0)]],
    device const float* inVector [[buffer(1)]],
    device float* output         [[buffer(2)]],
    constant uint& rows          [[buffer(3)]], // output elements
    constant uint& cols          [[buffer(4)]], // input elements
    uint index                   [[thread_position_in_grid]]
) {
    if (index >= rows) return;
    float result = 0.0f;
    for (uint k = 0; k < cols; k++) {
        result += inMatrix[index * cols + k] * inVector[k];
    }
    output[index] = result;

}

kernel void gemv_interleaved(
    device const float* inMatrix [[buffer(0)]],
    device const float* inVector [[buffer(1)]],
    device float* output         [[buffer(2)]],
    constant uint& rows          [[buffer(3)]], // output elements
    constant uint& cols          [[buffer(4)]], // input elements
    uint lane                    [[thread_index_in_simdgroup]],
    uint simd_width              [[threads_per_simdgroup]],
    uint2 tg                     [[threadgroup_position_in_grid]]
) {
    uint row = tg.x;
    if (row >= rows) return;

    float result = 0.0f;
    for (uint col = lane; col < cols; col += simd_width) {
        result = fma(inMatrix[row * cols + col], inVector[col], result);
    }
    
    result = simd_sum(result);
    if (lane == 0) {
        output[row] = result;
    }
}
