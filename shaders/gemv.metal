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
    uint index                   [[thread_position_in_grid]],
    uint lane                    [[thread_index_in_simdgroup]],
    uint tg_size                 [[threads_per_threadgroup]],
    uint tid                     [[thread_position_in_threadgroup]],
    uint simd_width              [[threads_per_simdgroup]],
    uint simdgroup_id            [[simdgroup_index_in_threadgroup]]
) {
    uint row = index;
    if (row >= rows) return;

    // uint groups_in_grid = (rows + simd_width - 1) / simd_width;
    uint group_id = index / simd_width;

    uint start_index = group_id * cols * simd_width + lane;
    uint end_index = start_index + simd_width * cols;

    float result = 0.0f;
    for (
        uint i = start_index, k = 0; 
        i < end_index; 
        i += simd_width, k++
    ) {
        result += inMatrix[i] * inVector[k];
    }
    output[index] = result;
}
