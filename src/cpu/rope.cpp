#include "llmetal/cpu/rope.hpp"
#include "llmetal/tensor.hpp"

namespace llmetal::cpu {

void rope_cos_and_sin(
    const std::uint32_t max_seq_length,
    const std::uint32_t rotary_dim,
    const float theta,
    llmetal::CpuTensor<float>& output // [max_seq_length, rotary_dim / 2, 2 (cos_and_sin)]
) {
    if (max_seq_length == 0) throw std::invalid_argument("max_seq_length must be non-zero");
    if (rotary_dim == 0 || rotary_dim % 2 != 0) throw std::invalid_argument("rotary_dim must be positive and even");
    if (!std::isfinite(theta) || theta <= 0.0f) throw std::invalid_argument("theta must be positive and finite");

    const std::uint32_t pair_count = rotary_dim / 2;
    const float rotary_dim_f = static_cast<float>(rotary_dim);

    if (output.shape().rank() != 3 ||
        output.shape()[0] != max_seq_length ||
        output.shape()[1] != pair_count ||
        output.shape()[2] != 2
    ) {
        throw std::invalid_argument("output must be a 3D tensor of shape [max_seq_length, rotary dim / 2, 2]");
    }

    for (std::uint32_t seq_idx = 0; seq_idx < max_seq_length; ++seq_idx) {
        const float seq_idx_f = static_cast<float>(seq_idx);

        for (std::uint32_t pair_idx = 0; pair_idx < pair_count; ++pair_idx) {
            const float pair_idx_f = static_cast<float>(pair_idx);

            const float inv_freq = std::powf(theta, -2.0f * pair_idx_f / rotary_dim_f);
            const float angle = inv_freq * seq_idx_f;

            output[seq_idx * pair_count * 2 + pair_idx * 2]      = std::cosf(angle);
            output[seq_idx * pair_count * 2 + pair_idx * 2 + 1 ] = std::sinf(angle);
        }
    }
}

void rotate_half(
    const llmetal::CpuTensor<float>& input,      // [batch, seq_length, num_heads, head_dim]
    llmetal::CpuTensor<float>& output,           // [batch, seq_length, num_heads, head_dim]
    const llmetal::CpuTensor<float>& cos_and_sin // [max_seq_length, rotary_dim / 2, 2 (cos_and_sin)]
) {
    if (input.shape().rank() != 4 || 
        output.shape().rank() != 4 || 
        input.shape()[0] != output.shape()[0] || 
        input.shape()[1] != output.shape()[1] || 
        input.shape()[2] != output.shape()[2] ||
        input.shape()[3] != output.shape()[3]
    ) {
        throw std::invalid_argument("input and output must be 4D tensors of the same shape");
    }

    if (cos_and_sin.shape().rank() != 3 || cos_and_sin.shape()[2] != 2) {
        throw std::invalid_argument("cos_and_sin must be a 3D tensor of shape [max_seq_length, rotary_dim / 2, 2]");
    }

    const std::uint32_t B = input.shape()[0]; // Batch size
    const std::uint32_t S = input.shape()[1]; // Sequence length
    const std::uint32_t H = input.shape()[2]; // Number of heads
    const std::uint32_t D = input.shape()[3]; // Head dimension
    const std::uint32_t P = cos_and_sin.shape()[1]; // Num rotary pairs
    const std::uint32_t R = P * 2; // Rotary dim

    if (R == 0) throw std::invalid_argument("rotary_dim must be positive");
    if (R > input.shape()[3]) throw std::invalid_argument("rotary_dim must be <= head_dim");
    if (cos_and_sin.shape()[0] != input.shape()[1]) throw std::invalid_argument("cos_and_sin max_seq_length must be longer than sequence length");

    for (std::size_t batch = 0; batch < B; ++batch) {
        for (std::size_t seq = 0; seq < S; ++seq) {
            for (std::size_t head = 0; head < H; ++head) {
                for (std::size_t pair = 0; pair < P; ++pair) {
                    std::size_t index = ((batch * S +  seq) * H + head) * D + pair;
                    
                    // Get rotation matrix values
                    const float c = cos_and_sin[seq * P * 2 + pair * 2];
                    const float s = cos_and_sin[seq * P * 2 + pair * 2 + 1];
                    
                    // Apply rotation (assignment to account for in-place substitution)
                    float pair_left_output =  input[index] * c - input[index + P] * s;
                    float pair_right_output = input[index] * s + input[index + P] * c;
                    output[index]     = pair_left_output;
                    output[index + P] = pair_right_output;
                }
                // Rotary dim may be less than head_dim, so copy over the rest
                for (std::size_t d = R; d < D; ++d) { 
                    std::size_t index =  ((batch * S + seq) * H + head) * D + d;
                    output[index] = input[index];
                }
            }
        }
    }
}

void rotate_half_in_place(
    llmetal::CpuTensor<float>& input,       // [batch, seq_length, num_heads, head_dim]
    const llmetal::CpuTensor<float>& cos_and_sin // [max_seq_length, rotary_dim / 2, 2 (cos_and_sin)]
) {
    rotate_half(input, input, cos_and_sin);    
}

} // namespace llmetal::cpu
