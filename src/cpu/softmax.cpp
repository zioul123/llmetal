#include "llmetal/cpu/softmax.hpp"

namespace llmetal::cpu {

void softmax(
    const CpuTensor<float>& logits,        // [rows, seq_length]
    const CpuTensor<std::uint32_t>& valid, // [rows]
    CpuTensor<float>& probs                // [rows, seq_length]
) {
    if (logits.shape().rank() != 2) throw std::invalid_argument("Invalid logits shape rank");
    if (valid.shape().rank() != 1) throw std::invalid_argument("Invalid valid shape rank");
    if (probs.shape().rank() != 2) throw std::invalid_argument("Invalid probs shape rank");

    std::uint32_t rows = checked_u32(logits.shape()[0], "softmax_rows");
    std::uint32_t seq_length = checked_u32(logits.shape()[1], "softmax_seq_length");
    if (valid.shape()[0] != rows) throw std::invalid_argument("Invalid valid shape");
    if (probs.numel() != logits.numel()) throw std::invalid_argument("Invalid probs shape - expected same size as logits");

    for (std::size_t r = 0; r < rows; ++r) {
        std::size_t row_offset = r * seq_length;
        std::size_t valid_seq = valid[r];

        // Get the max
        float r_max = -std::numeric_limits<float>::infinity();
        for (std::size_t s = 0; s < valid_seq; ++s) {
            r_max = std::max(r_max, logits[row_offset + s]);
        }

        // Exponentiate
        float factor = 0.0f;
        for (std::size_t s = 0; s < valid_seq; ++s) {
            std::size_t r_c = row_offset + s;
            float curr_val = std::exp(logits[r_c] - r_max);
            probs[row_offset + s] = curr_val;
            factor += curr_val;
        }

        // Normalize
        factor = 1.0f / factor; // Invert for multiplication
        for (std::size_t s = 0; s < valid_seq; ++s) {
            probs[row_offset + s] *= factor;
        }
    }
}

} // namespace llmetal::cpu
