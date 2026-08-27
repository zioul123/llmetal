#include "llmetal/cpu/softmax.hpp"

namespace llmetal::cpu {

void softmax(
    const CpuTensor<float>& logits,        // [rows,C]
    const CpuTensor<std::uint32_t>& valid, // [rows]
    CpuTensor<float>& probs                // [rows,C]
) {
    if (logits.shape().rank() != 2) throw std::invalid_argument("Invalid logits shape rank");
    if (valid.shape().rank() != 1) throw std::invalid_argument("Invalid valid shape rank");
    if (probs.shape().rank() != 2) throw std::invalid_argument("Invalid probs shape rank");

    std::uint32_t R = checked_u32(logits.shape()[0], "softmax_rows");
    std::uint32_t C = checked_u32(logits.shape()[1], "softmax_rows");
    if (valid.shape()[0] != R) throw std::invalid_argument("Invalid valid shape");
    if (probs.numel() != logits.numel()) throw std::invalid_argument("Invalid probs shape - expected same size as logits");

    for (std::size_t r = 0; r < R; ++r) {
        std::size_t row_offset = r * C;
        std::size_t valid_c = valid[r];

        // Get the max
        float r_max = -std::numeric_limits<float>::infinity();
        for (std::size_t c = 0; c < valid_c; ++c) {
            r_max = std::max(r_max, logits[row_offset + c]);
        }

        // Exponentiate
        float factor = 0.0f;
        for (std::size_t c = 0; c < valid_c; ++c) {
            std::size_t r_c = row_offset + c;
            float curr_val = std::exp(logits[r_c] - r_max);
            probs[row_offset + c] = curr_val;
            factor += curr_val;
        }

        // Normalize
        factor = 1.0f / factor; // Invert for multiplication
        for (std::size_t c = 0; c < valid_c; ++c) {
            probs[row_offset + c] *= factor;
        }
    }
}

} // namespace llmetal::cpu
