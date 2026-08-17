#include <metal_stdlib>
#include <metal_simdgroup>

using namespace metal;

kernel void linear_naive_with_bias(
    device const float* input         [[buffer(0)]], // [batch_size, sequence_length, input_hidden_size]
    device const float* weight        [[buffer(1)]], // [output_hidden_size, input_hidden_size]
    device const float* bias          [[buffer(2)]], // [output_hidden_size]
    device float* output              [[buffer(3)]], // [batch_size, sequence_length, output_hidden_size]
    constant uint& input_hidden_size  [[buffer(4)]],
    constant uint& output_hidden_size [[buffer(5)]],
    constant uint& sequence_length    [[buffer(6)]],
    uint2 index                       [[thread_position_in_grid]], // index.x: output_row, index.y: batch and sequence
    uint2 grid_size                   [[threads_per_grid]]
) {
    if (index.y >= grid_size.y) return;

    uint o = index.x;
    uint b = index.y / sequence_length;
    uint s = index.y % sequence_length;
    uint bs_in_offset = (b * sequence_length + s) * input_hidden_size;
    uint bs_out_offset = (b * sequence_length + s) * output_hidden_size;
    uint w_offset = o * input_hidden_size;

    float sum = 0.0f;
    for (uint i = 0; i < input_hidden_size; ++i) { 
        sum = fma(weight[w_offset + i], input[bs_in_offset + i], sum);
    }
    output[bs_out_offset + o] = sum + bias[o];
}

// Only 1 simd group per row, multiple rows per threadgroup
kernel void linear_with_bias_1sgpr(
    device const float* input         [[buffer(0)]], // [batch_size, sequence_length, input_hidden_size]
    device const float* weight        [[buffer(1)]], // [output_hidden_size, input_hidden_size]
    device const float* bias          [[buffer(2)]], // [output_hidden_size]
    device float* output              [[buffer(3)]], // [batch_size, sequence_length, output_hidden_size]
    constant uint& input_hidden_size  [[buffer(4)]],
    constant uint& output_hidden_size [[buffer(5)]],
    constant uint& sequence_length    [[buffer(6)]],
    uint2 index                       [[thread_position_in_grid]], // index.x: lane, index.y: row
    uint2 grid_size                   [[threads_per_grid]],
    uint tpsg                         [[threads_per_simdgroup]]
) {
    // if (index.y >= grid_size.y) return;

    // uint o = index.x;
    // uint b = index.y / sequence_length;
    // uint s = index.y % sequence_length;
    // uint bs_in_offset = (b * sequence_length + s) * input_hidden_size;
    // uint bs_out_offset = (b * sequence_length + s) * output_hidden_size;
    // uint w_offset = o * input_hidden_size;

    // float sum = 0.0f;
    // for (uint i = 0; i < input_hidden_size; ++i) { 
    //     sum = fma(weight[w_offset + i], input[bs_in_offset + i], sum);
    // }
    // output[bs_out_offset + o] = sum + bias[o];
}

// N simd groups per row, one row per threadgroup
kernel void linear_with_bias_nsgpr(
    device const float* input         [[buffer(0)]], // [batch_size, sequence_length, input_hidden_size]
    device const float* weight        [[buffer(1)]], // [output_hidden_size, input_hidden_size]
    device const float* bias          [[buffer(2)]], // [output_hidden_size]
    device float* output              [[buffer(3)]], // [batch_size, sequence_length, output_hidden_size]
    constant uint& input_hidden_size  [[buffer(4)]],
    constant uint& output_hidden_size [[buffer(5)]],
    constant uint& sequence_length    [[buffer(6)]],
    constant uint& sgptg              [[buffer(7)]],
    uint2 index                       [[thread_position_in_grid]], // index.x: lane, index.y: row
    uint2 grid_size                   [[threads_per_grid]],
    uint tpsg                         [[threads_per_simdgroup]],
    uint lane                         [[thread_index_in_simdgroup]]
) {
    // if (index.y >= grid_size.y) return;

    // uint o = index.x;
    // uint b = index.y / sequence_length;
    // uint s = index.y % sequence_length;
    // uint bs_in_offset = (b * sequence_length + s) * input_hidden_size;
    // uint bs_out_offset = (b * sequence_length + s) * output_hidden_size;
    // uint w_offset = o * input_hidden_size;

    // float sum = 0.0f;
    // for (uint i = 0; i < input_hidden_size; ++i) { 
    //     sum = fma(weight[w_offset + i], input[bs_in_offset + i], sum);
    // }
    // output[bs_out_offset + o] = sum + bias[o];
}

kernel void linear_naive_without_bias(
    device const float* input         [[buffer(0)]], // [batch_size, sequence_length, input_hidden_size]
    device const float* weight        [[buffer(1)]], // [output_hidden_size, input_hidden_size]
    device float* output              [[buffer(2)]], // [batch_size, sequence_length, output_hidden_size]
    constant uint& input_hidden_size  [[buffer(3)]],
    constant uint& output_hidden_size [[buffer(4)]],
    constant uint& sequence_length    [[buffer(5)]],
    uint2 index                       [[thread_position_in_grid]], // index.x: output_row, index.y: batch and sequence
    uint2 grid_size                   [[threads_per_grid]]
) {
    if (index.y >= grid_size.y) return;

    uint o = index.x;
    uint b = index.y / sequence_length;
    uint s = index.y % sequence_length;
    uint bs_in_offset = (b * sequence_length + s) * input_hidden_size;
    uint bs_out_offset = (b * sequence_length + s) * output_hidden_size;
    uint w_offset = o * input_hidden_size;

    float sum = 0.0f;
    for (uint i = 0; i < input_hidden_size; ++i) { 
        sum = fma(weight[w_offset + i], input[bs_in_offset + i], sum);
    }
    output[bs_out_offset + o] = sum;
}

// Only 1 simd group per row, multiple rows per threadgroup
kernel void linear_without_bias_1sgpr(
    device const float* input         [[buffer(0)]], // [batch_size, sequence_length, input_hidden_size]
    device const float* weight        [[buffer(1)]], // [output_hidden_size, input_hidden_size]
    device float* output              [[buffer(2)]], // [batch_size, sequence_length, output_hidden_size]
    constant uint& input_hidden_size  [[buffer(3)]],
    constant uint& output_hidden_size [[buffer(4)]],
    constant uint& sequence_length    [[buffer(5)]],
    uint2 index                       [[thread_position_in_grid]], // index.x: lane, index.y: row
    uint2 grid_size                   [[threads_per_grid]],
    uint tpsg                         [[threads_per_simdgroup]]
) {
    // if (index.y >= grid_size.y) return;

    // uint o = index.x;
    // uint b = index.y / sequence_length;
    // uint s = index.y % sequence_length;
    // uint bs_in_offset = (b * sequence_length + s) * input_hidden_size;
    // uint bs_out_offset = (b * sequence_length + s) * output_hidden_size;
    // uint w_offset = o * input_hidden_size;

    // float sum = 0.0f;
    // for (uint i = 0; i < input_hidden_size; ++i) { 
    //     sum = fma(weight[w_offset + i], input[bs_in_offset + i], sum);
    // }
    // output[bs_out_offset + o] = sum + bias[o];
}

// N simd groups per row, one row per threadgroup
kernel void linear_without_bias_nsgpr(
    device const float* input         [[buffer(0)]], // [batch_size, sequence_length, input_hidden_size]
    device const float* weight        [[buffer(1)]], // [output_hidden_size, input_hidden_size]
    device float* output              [[buffer(2)]], // [batch_size, sequence_length, output_hidden_size]
    constant uint& input_hidden_size  [[buffer(3)]],
    constant uint& output_hidden_size [[buffer(4)]],
    constant uint& sequence_length    [[buffer(5)]],
    constant uint& sgptg              [[buffer(6)]],
    uint2 index                       [[thread_position_in_grid]], // index.x: lane, index.y: row
    uint2 grid_size                   [[threads_per_grid]],
    uint tpsg                         [[threads_per_simdgroup]],
    uint lane                         [[thread_index_in_simdgroup]]
) {
    // if (index.y >= grid_size.y) return;

    // uint o = index.x;
    // uint b = index.y / sequence_length;
    // uint s = index.y % sequence_length;
    // uint bs_in_offset = (b * sequence_length + s) * input_hidden_size;
    // uint bs_out_offset = (b * sequence_length + s) * output_hidden_size;
    // uint w_offset = o * input_hidden_size;

    // float sum = 0.0f;
    // for (uint i = 0; i < input_hidden_size; ++i) { 
    //     sum = fma(weight[w_offset + i], input[bs_in_offset + i], sum);
    // }
    // output[bs_out_offset + o] = sum + bias[o];
}
