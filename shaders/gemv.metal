#include <metal_stdlib>

using namespace metal;

kernel void my_gemv(
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
