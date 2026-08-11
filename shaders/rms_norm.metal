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

// N simd groups per row, one row per threadgroup
kernel void rms_norm_nsgpr(
    device const float* input    [[buffer(0)]], // [batch_size, sequence_length, hidden_size]
    device const float* weights  [[buffer(1)]], // [hidden_size]
    device float* output         [[buffer(2)]], // [batch_size, sequence_length, hidden_size]
    constant uint& hidden_size   [[buffer(3)]],
    constant float& epsilon      [[buffer(4)]],
    constant uint& sgpr          [[buffer(5)]],
    uint2 index                  [[thread_position_in_grid]], // index.x: lane, index.y: row
    uint2 grid_size              [[threads_per_grid]],
    uint tpsg                    [[threads_per_simdgroup]],
    uint lane                    [[thread_index_in_simdgroup]]
) {
    threadgroup float partial_sums[32]; // Max 32 simd groups per row. A bit too much though.

    if (index.y >= grid_size.y) return;

    // columns per simd group is ceil_div(hidden_size, sgpr)
    uint cpsg = (hidden_size + sgpr - 1) / sgpr;
    // the simd group index (per row) is thread index / simd group size
    uint sg_idx = index.x / tpsg;

    // Input/output row offset
    uint row_idx = index.y * hidden_size;
    // Simd group column bounds
    uint sg_start = sg_idx * cpsg;
    uint sg_end = min(sg_start + cpsg, hidden_size);
    
    float sum = 0.0f;
    for (uint h = sg_start + lane; h < sg_end; h += tpsg) {
        float curr = input[row_idx + h];
        sum += curr * curr;
    }
    
    // Sum within each simd group and let lane 0 store it
    sum = simd_sum(sum);
    if (lane == 0) {
        partial_sums[sg_idx] = sum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Let simd group 0 sum across simd groups and compute inv rms
    float total = 0.0f;
    if (sg_idx == 0)
    {
        total = lane < sgpr ? partial_sums[lane] : 0.0f;
        total = simd_sum(total);
        if (lane == 0) {
            partial_sums[0] = rsqrt(total / hidden_size + epsilon);;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    // Grab it
    float inv_rms = partial_sums[0];

    for (uint h = sg_start + lane; h < sg_end; h += tpsg) {
        output[row_idx + h] = input[row_idx + h] * inv_rms * weights[h];
    }
}
