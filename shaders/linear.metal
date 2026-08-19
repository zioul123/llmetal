#include <metal_stdlib>
#include <metal_simdgroup>

using namespace metal;

constant bool has_bias [[function_constant(0)]];

kernel void linear_naive(
    device const float* input         [[buffer(0)]], // [batch_size, sequence_length, input_hidden_size]
    device const float* weight        [[buffer(1)]], // [output_hidden_size, input_hidden_size]
    device const float* bias          [[buffer(2), function_constant(has_bias)]], // [output_hidden_size]
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
    output[bs_out_offset + o] = has_bias ? sum + bias[o] : sum;
}

// Only 1 simd group per row, multiple rows per threadgroup
kernel void linear_1sgpr(
    device const float* input         [[buffer(0)]], // [batch_size, sequence_length, input_hidden_size]
    device const float* weight        [[buffer(1)]], // [output_hidden_size, input_hidden_size]
    device const float* bias          [[buffer(2), function_constant(has_bias)]], // [output_hidden_size]
    device float* output              [[buffer(3)]], // [batch_size, sequence_length, output_hidden_size]
    constant uint& input_hidden_size  [[buffer(4)]],
    constant uint& output_hidden_size [[buffer(5)]],
    constant uint& sequence_length    [[buffer(6)]],
    uint3 index                       [[thread_position_in_grid]], // index.x: lane, index.y: output_row, index.z: batch and sequence
    uint3 grid_size                   [[threads_per_grid]],
    uint tpsg                         [[threads_per_simdgroup]]
) {
    if (index.x >= input_hidden_size || index.y >= grid_size.y || index.z >= grid_size.z) return;

    uint o = index.y;
    uint b = index.z / sequence_length;
    uint s = index.z % sequence_length;
    uint bs_in_offset = (b * sequence_length + s) * input_hidden_size;
    uint bs_out_offset = (b * sequence_length + s) * output_hidden_size;
    uint w_offset = o * input_hidden_size;

    float sum = 0.0f;
    for (uint i = index.x; i < input_hidden_size; i += tpsg) { 
        sum = fma(weight[w_offset + i], input[bs_in_offset + i], sum);
    }
    sum = simd_sum(sum);
    if (index.x == 0) {
        output[bs_out_offset + o] = has_bias ? sum + bias[o] : sum;
    }
}

// N simd groups per row, one row per threadgroup
kernel void linear_nsgpr(
    device const float* input         [[buffer(0)]], // [batch_size, sequence_length, input_hidden_size]
    device const float* weight        [[buffer(1)]], // [output_hidden_size, input_hidden_size]
    device const float* bias          [[buffer(2), function_constant(has_bias)]], // [output_hidden_size]
    device float* output              [[buffer(3)]], // [batch_size, sequence_length, output_hidden_size]
    constant uint& input_hidden_size  [[buffer(4)]],
    constant uint& output_hidden_size [[buffer(5)]],
    constant uint& sequence_length    [[buffer(6)]],
    constant uint& sgpr               [[buffer(7)]],
    uint3 index                       [[thread_position_in_grid]], // index.x: lane, index.y: row
    uint3 grid_size                   [[threads_per_grid]],
    uint tpsg                         [[threads_per_simdgroup]],
    uint lane                         [[thread_index_in_simdgroup]]
) {
    threadgroup float partial_sums[32]; // Max 32 simd groups per row. A bit too much though.

    if (index.y >= grid_size.y || index.z >= grid_size.z) return;

    uint cpsg = (input_hidden_size + sgpr - 1) / sgpr;
    uint o = index.y;
    uint b = index.z / sequence_length;
    uint s = index.z % sequence_length;

    // Get row related information
    uint bs_in_offset = (b * sequence_length + s) * input_hidden_size;
    uint bs_out_offset = (b * sequence_length + s) * output_hidden_size;
    uint w_offset = o * input_hidden_size;

    // Get column related information
    uint sg_idx = index.x / tpsg;
    uint sg_start = sg_idx * cpsg;
    uint sg_end = min(sg_start + cpsg, input_hidden_size);

    float sum = 0.0f;
    for (uint i = sg_start + lane; i < sg_end; i += tpsg) { 
        sum = fma(weight[w_offset + i], input[bs_in_offset + i], sum);
    }
    sum = simd_sum(sum);
    if (lane == 0) {
        partial_sums[sg_idx] = sum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg_idx == 0) {
        float total = lane < sgpr ? partial_sums[lane] : 0.0f;
        total = simd_sum(total);
        // total = simd_sum(partial_sums[lane]);
        if (lane == 0) {
            output[bs_out_offset + o] = has_bias ? total + bias[o] : total;
        }
    }
}

template<uint ROWS_PER_THREAD>
kernel void linear_naive_x(
    device const float* input         [[buffer(0)]], // [batch_size, sequence_length, input_hidden_size]
    device const float* weight        [[buffer(1)]], // [output_hidden_size, input_hidden_size]
    device const float* bias          [[buffer(2), function_constant(has_bias)]], // [output_hidden_size]
    device float* output              [[buffer(3)]], // [batch_size, sequence_length, output_hidden_size]
    constant uint& input_hidden_size  [[buffer(4)]],
    constant uint& output_hidden_size [[buffer(5)]],
    constant uint& sequence_length    [[buffer(6)]],
    uint2 index                       [[thread_position_in_grid]], // index.x: output_row_group, index.y: batch and sequence
    uint2 grid_size                   [[threads_per_grid]]
) {
    uint o = index.x * ROWS_PER_THREAD;
    uint b = index.y / sequence_length;
    uint s = index.y % sequence_length;
    uint bs_in_offset = (b * sequence_length + s) * input_hidden_size;
    uint bs_out_offset = (b * sequence_length + s) * output_hidden_size;
    uint w_offset = o * input_hidden_size;

    float acc[ROWS_PER_THREAD] = {0.0f};
    
        
    // Two loops are duplicates, but the first is for when it all rows are to be used,
    // the second is when it's not a nice multiple and we need to trim the final rows
    if (o + ROWS_PER_THREAD <= output_hidden_size) {
        for (uint i = 0; i < input_hidden_size; ++i) { 
            float vecValue = input[bs_in_offset + i];
            #pragma unroll
            for (uint r = 0; r < ROWS_PER_THREAD; ++r) {
                acc[r] = fma(weight[w_offset + r * input_hidden_size + i], vecValue, acc[r]);
            }
        }
    } else {
        for (uint i = 0; i < input_hidden_size; ++i) { 
            float vecValue = input[bs_in_offset + i];
            #pragma unroll
            for (uint r = 0; r < ROWS_PER_THREAD; ++r) {
                if (o + r < output_hidden_size) {
                    acc[r] = fma(weight[w_offset + r * input_hidden_size + i], vecValue, acc[r]);
                }
            }
        }
    }

    #pragma unroll
    for (uint r = 0; r < ROWS_PER_THREAD; ++r) {
        if (o + r < output_hidden_size) {
            output[bs_out_offset + o + r] = has_bias ? acc[r] + bias[o + r] : acc[r];
        }
    }
}

template [[host_name("linear_naive_x2")]]
kernel void linear_naive_x<2>(
    device const float*, device const float*, device const float*,
    device float*,       constant uint&,      constant uint&,
    constant uint&,      uint2,               uint2
);

template [[host_name("linear_naive_x4")]]
kernel void linear_naive_x<4>(
    device const float*, device const float*, device const float*,
    device float*,       constant uint&,      constant uint&,
    constant uint&,      uint2,               uint2
);

template [[host_name("linear_naive_x8")]]
kernel void linear_naive_x<8>(
    device const float*, device const float*, device const float*,
    device float*,       constant uint&,      constant uint&,
    constant uint&,      uint2,               uint2
);

// Only 1 simd group per row, multiple rows per threadgroup
template<uint ROWS_PER_THREAD>
kernel void linear_1sgpr_x(
    device const float* input         [[buffer(0)]], // [batch_size, sequence_length, input_hidden_size]
    device const float* weight        [[buffer(1)]], // [output_hidden_size, input_hidden_size]
    device const float* bias          [[buffer(2), function_constant(has_bias)]], // [output_hidden_size]
    device float* output              [[buffer(3)]], // [batch_size, sequence_length, output_hidden_size]
    constant uint& input_hidden_size  [[buffer(4)]],
    constant uint& output_hidden_size [[buffer(5)]],
    constant uint& sequence_length    [[buffer(6)]],
    uint3 index                       [[thread_position_in_grid]], // index.x: lane, index.y: output_row, index.z: batch and sequence
    uint3 grid_size                   [[threads_per_grid]],
    uint tpsg                         [[threads_per_simdgroup]]
) {
    if (index.x >= input_hidden_size || index.y >= grid_size.y || index.z >= grid_size.z) return;

    uint o = index.y * ROWS_PER_THREAD;
    uint b = index.z / sequence_length;
    uint s = index.z % sequence_length;
    uint bs_in_offset = (b * sequence_length + s) * input_hidden_size;
    uint bs_out_offset = (b * sequence_length + s) * output_hidden_size;
    uint w_offset = o * input_hidden_size;

    float acc[ROWS_PER_THREAD] = {0.0f};
    // Two loops are duplicates, but the first is for when it all rows are to be used,
    // the second is when it's not a nice multiple and we need to trim the final rows
    if (o + ROWS_PER_THREAD <= output_hidden_size) {
        for (uint i = index.x; i < input_hidden_size; i += tpsg) { 
            float vecValue = input[bs_in_offset + i];
            #pragma unroll
            for (uint r = 0; r < ROWS_PER_THREAD; ++r) {
                acc[r] = fma(weight[w_offset + r * input_hidden_size + i], vecValue, acc[r]);
            }
        }
    } else { 
        for (uint i = index.x; i < input_hidden_size; i += tpsg) { 
            float vecValue = input[bs_in_offset + i];
            #pragma unroll
            for (uint r = 0; r < ROWS_PER_THREAD; ++r) {
                if (o + r < output_hidden_size) {
                    acc[r] = fma(weight[w_offset + r * input_hidden_size +i], vecValue, acc[r]);
                }
            }
        }
    }
    #pragma unroll
    for (uint r = 0; r < ROWS_PER_THREAD; ++r) {
        acc[r] = simd_sum(acc[r]);
    }
    if (index.x == 0) {
        #pragma unroll
        for (uint r = 0; r < ROWS_PER_THREAD; ++r) {
            if (o + r < output_hidden_size) {
                output[bs_out_offset + o + r] = has_bias ? acc[r] + bias[o + r] : acc[r];
            }
        }
    }
}

template [[host_name("linear_1sgpr_x2")]]
kernel void linear_1sgpr_x<2>(
    device const float*, device const float*, device const float*,
    device float*,       constant uint&,      constant uint&,
    constant uint&,      uint3,               uint3,
    uint
);

template [[host_name("linear_1sgpr_x4")]]
kernel void linear_1sgpr_x<4>(
    device const float*, device const float*, device const float*,
    device float*,       constant uint&,      constant uint&,
    constant uint&,      uint3,               uint3,
    uint
);

template [[host_name("linear_1sgpr_x8")]]
kernel void linear_1sgpr_x<8>(
    device const float*, device const float*, device const float*,
    device float*,       constant uint&,      constant uint&,
    constant uint&,      uint3,               uint3,
    uint
);

// N simd groups per row, one row per threadgroup
kernel void linear_nsgpr_x(
    device const float* input         [[buffer(0)]], // [batch_size, sequence_length, input_hidden_size]
    device const float* weight        [[buffer(1)]], // [output_hidden_size, input_hidden_size]
    device const float* bias          [[buffer(2), function_constant(has_bias)]], // [output_hidden_size]
    device float* output              [[buffer(3)]], // [batch_size, sequence_length, output_hidden_size]
    constant uint& input_hidden_size  [[buffer(4)]],
    constant uint& output_hidden_size [[buffer(5)]],
    constant uint& sequence_length    [[buffer(6)]],
    constant uint& sgpr               [[buffer(7)]],
    uint3 index                       [[thread_position_in_grid]], // index.x: lane, index.y: row
    uint3 grid_size                   [[threads_per_grid]],
    uint tpsg                         [[threads_per_simdgroup]],
    uint lane                         [[thread_index_in_simdgroup]]
) {
    threadgroup float partial_sums[32]; // Max 32 simd groups per row. A bit too much though.

    if (index.y >= grid_size.y || index.z >= grid_size.z) return;

    uint cpsg = (input_hidden_size + sgpr - 1) / sgpr;
    uint o = index.y;
    uint b = index.z / sequence_length;
    uint s = index.z % sequence_length;

    // Get row related information
    uint bs_in_offset = (b * sequence_length + s) * input_hidden_size;
    uint bs_out_offset = (b * sequence_length + s) * output_hidden_size;
    uint w_offset = o * input_hidden_size;

    // Get column related information
    uint sg_idx = index.x / tpsg;
    uint sg_start = sg_idx * cpsg;
    uint sg_end = min(sg_start + cpsg, input_hidden_size);

    float sum = 0.0f;
    for (uint i = sg_start + lane; i < sg_end; i += tpsg) { 
        sum = fma(weight[w_offset + i], input[bs_in_offset + i], sum);
    }
    sum = simd_sum(sum);
    if (lane == 0) {
        partial_sums[sg_idx] = sum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg_idx == 0) {
        float total = lane < sgpr ? partial_sums[lane] : 0.0f;
        total = simd_sum(total);
        // total = simd_sum(partial_sums[lane]);
        if (lane == 0) {
            output[bs_out_offset + o] = has_bias ? total + bias[o] : total;
        }
    }
}
