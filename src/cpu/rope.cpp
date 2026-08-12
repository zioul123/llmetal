#include "llmetal/cpu/rope.hpp"

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

    if (output.shape().rank() != 3 ||
        output.shape()[0] != max_seq_length ||
        output.shape()[1] != pair_count ||
        output.shape()[2] != 2
    ) {
        throw std::invalid_argument("output must be a 3D tensor of shape [max_seq_length, rotary dim / 2, 2]");
    }

    for (std::uint32_t seq_idx = 0; seq_idx < max_seq_length; ++seq_idx) {
        for (std::uint32_t pair_idx = 0; pair_idx < pair_count; ++pair_idx) {
            const float inv_freq = std::powf(
                theta, 
                -2.0f * static_cast<float>(pair_idx) / static_cast<float>(rotary_dim)
            );
            const float angle = inv_freq * static_cast<float>(seq_idx);
            output[seq_idx * pair_count * 2 + pair_idx * 2] = std::cosf(angle);
            output[seq_idx * pair_count * 2 + pair_idx * 2 + 1 ] = std::sinf(angle);
        }
    }
}

} // namespace llmetal::cpu
