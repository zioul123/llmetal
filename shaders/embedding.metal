#include <metal_stdlib>

using namespace metal;

kernel void embedding_naive(
    device const float* table    [[buffer(0)]], // [vocab_size, hidden_size]
    device const uint* inVector  [[buffer(1)]], // [batch_size, sequence_length]
    device float* output         [[buffer(2)]], // [batch_size, sequence_length, hidden_size]
    constant uint& hidden_size   [[buffer(3)]],
    constant uint& vocab_size    [[buffer(4)]],
    uint index                   [[thread_position_in_grid]],
    uint grid_size               [[threads_per_grid]]
) {
    if (index >= grid_size) return;

    uint token_id = inVector[index];
    if (token_id >= vocab_size) return; 
    
    uint input_offset = token_id * hidden_size;
    uint output_offset = index * hidden_size;
    for (uint h = 0; h < hidden_size; ++h) {
        output[output_offset + h] = table[input_offset + h];
    }
}
