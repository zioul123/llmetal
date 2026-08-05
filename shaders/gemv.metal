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

// === N Rows per SIMD group, 4 unrolled variants ===

kernel void gemv_1RPSG(
    device const float* inMatrix [[buffer(0)]],
    device const float* inVector [[buffer(1)]],
    device float* output         [[buffer(2)]],
    constant uint& rows          [[buffer(3)]], // output elements
    constant uint& cols          [[buffer(4)]], // input elements
    uint lane                    [[thread_index_in_simdgroup]],
    uint sg                      [[simdgroup_index_in_threadgroup]],
    uint sg_size                 [[threads_per_simdgroup]],
    uint2 tg                     [[threadgroup_position_in_grid]],
    uint2 tg_size                [[threads_per_threadgroup]]
) {
    uint sgptg = tg_size.x / sg_size;
    uint row = tg.x * sgptg + sg;
    if (row >= rows) return;

    float result = 0.0f;
    for (uint col = lane; col < cols; col += sg_size) {
        result = fma(inMatrix[row * cols + col], inVector[col], result);
    }
    
    result = simd_sum(result);
    if (lane == 0) {
        output[row] = result;
    }
}

kernel void gemv_2RPSG(
    device const float* inMatrix [[buffer(0)]],
    device const float* inVector [[buffer(1)]],
    device float* output         [[buffer(2)]],
    constant uint& rows          [[buffer(3)]], // output elements
    constant uint& cols          [[buffer(4)]], // input elements
    uint lane                    [[thread_index_in_simdgroup]],
    uint sg                      [[simdgroup_index_in_threadgroup]],
    uint sg_size                 [[threads_per_simdgroup]],
    uint2 tg                     [[threadgroup_position_in_grid]],
    uint2 tg_size                [[threads_per_threadgroup]]
) {
    uint sgptg = tg_size.x / sg_size;
    uint row = tg.x * 2 * sgptg + sg * 2;
    if (row >= rows) return;

    float result0 = 0.0f;
    float result1 = 0.0f;

    // Get more juice out of each column read
    // Increase instruction parallelism
    for (uint col = lane; col < cols; col += sg_size) {
        float vecValue = inVector[col];
        result0 = fma(inMatrix[ row      * cols + col], vecValue, result0);
        result1 = fma(inMatrix[(row + 1) * cols + col], vecValue, result1);
    }
    
    result0 = simd_sum(result0);
    result1 = simd_sum(result1);

    if (lane == 0) {
        output[row    ] = result0;
        output[row + 1] = result1;
    }
}

kernel void gemv_4RPSG(
    device const float* inMatrix [[buffer(0)]],
    device const float* inVector [[buffer(1)]],
    device float* output         [[buffer(2)]],
    constant uint& rows          [[buffer(3)]], // output elements
    constant uint& cols          [[buffer(4)]], // input elements
    uint lane                    [[thread_index_in_simdgroup]],
    uint sg                      [[simdgroup_index_in_threadgroup]],
    uint sg_size                 [[threads_per_simdgroup]],
    uint2 tg                     [[threadgroup_position_in_grid]],
    uint2 tg_size                [[threads_per_threadgroup]]
) {
    uint sgptg = tg_size.x / sg_size;
    uint row = tg.x * 4 * sgptg + sg * 4;
    if (row >= rows) return;

    float result0 = 0.0f;
    float result1 = 0.0f;
    float result2 = 0.0f;
    float result3 = 0.0f;

    // Get more juice out of each column read
    // Increase instruction parallelism
    for (uint col = lane; col < cols; col += sg_size) {
        float vecValue = inVector[col];
        result0 = fma(inMatrix[ row      * cols + col], vecValue, result0);
        result1 = fma(inMatrix[(row + 1) * cols + col], vecValue, result1);
        result2 = fma(inMatrix[(row + 2) * cols + col], vecValue, result2);
        result3 = fma(inMatrix[(row + 3) * cols + col], vecValue, result3);
    }
    
    result0 = simd_sum(result0);
    result1 = simd_sum(result1);
    result2 = simd_sum(result2);
    result3 = simd_sum(result3);

    if (lane == 0) {
        output[row    ] = result0;
        output[row + 1] = result1;
        output[row + 2] = result2;
        output[row + 3] = result3;
    }
}

kernel void gemv_8RPSG(
    device const float* inMatrix [[buffer(0)]],
    device const float* inVector [[buffer(1)]],
    device float* output         [[buffer(2)]],
    constant uint& rows          [[buffer(3)]], // output elements
    constant uint& cols          [[buffer(4)]], // input elements
    uint lane                    [[thread_index_in_simdgroup]],
    uint sg                      [[simdgroup_index_in_threadgroup]],
    uint sg_size                 [[threads_per_simdgroup]],
    uint2 tg                     [[threadgroup_position_in_grid]],
    uint2 tg_size                [[threads_per_threadgroup]]
) {
    uint sgptg = tg_size.x / sg_size;
    uint row = tg.x * 8 * sgptg + sg * 8;
    if (row >= rows) return;

    float result0 = 0.0f;
    float result1 = 0.0f;
    float result2 = 0.0f;
    float result3 = 0.0f;
    float result4 = 0.0f;
    float result5 = 0.0f;
    float result6 = 0.0f;
    float result7 = 0.0f;

    // Get more juice out of each column read
    // Increase instruction parallelism
    for (uint col = lane; col < cols; col += sg_size) {
        float vecValue = inVector[col];
        result0 = fma(inMatrix[ row      * cols + col], vecValue, result0);
        result1 = fma(inMatrix[(row + 1) * cols + col], vecValue, result1);
        result2 = fma(inMatrix[(row + 2) * cols + col], vecValue, result2);
        result3 = fma(inMatrix[(row + 3) * cols + col], vecValue, result3);
        result4 = fma(inMatrix[(row + 4) * cols + col], vecValue, result4);
        result5 = fma(inMatrix[(row + 5) * cols + col], vecValue, result5);
        result6 = fma(inMatrix[(row + 6) * cols + col], vecValue, result6);
        result7 = fma(inMatrix[(row + 7) * cols + col], vecValue, result7);
    }
    
    result0 = simd_sum(result0);
    result1 = simd_sum(result1);
    result2 = simd_sum(result2);
    result3 = simd_sum(result3);
    result4 = simd_sum(result4);
    result5 = simd_sum(result5);
    result6 = simd_sum(result6);
    result7 = simd_sum(result7);

    if (lane == 0) {
        output[row    ] = result0;
        output[row + 1] = result1;
        output[row + 2] = result2;
        output[row + 3] = result3;
        output[row + 4] = result4;
        output[row + 5] = result5;
        output[row + 6] = result6;
        output[row + 7] = result7;
    }
}
