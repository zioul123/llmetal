#include <cstddef>

#include "llmetal/tensor.hpp"

namespace llmetal::cpu {

void rms_norm(
    const llmetal::CpuTensor<float>& input,   // [batch_size, sequence_length, hidden]
    const llmetal::CpuTensor<float>& weights, // [hidden]
    llmetal::CpuTensor<float>& output,        // [batch_size, sequence_length, hidden]
    const float epsilon
) {
    if (input.shape().rank() != 3) throw std::invalid_argument("Invalid input shape rank");
    if (weights.shape().rank() != 1) throw std::invalid_argument("Invalid weights shape rank");
    if (output.shape().rank() != 3) throw std::invalid_argument("Invalid output shape rank");

    std::uint32_t batch_size = checked_u32(input.shape()[0], "cpu_rms_norm_batch_size");
    std::uint32_t sequence_length = checked_u32(input.shape()[1], "cpu_rms_norm_sequence_length");
    std::uint32_t hidden = checked_u32(weights.shape()[0], "cpu_rms_norm_hidden");

    if (input.shape()[2] != hidden) throw std::invalid_argument("Invalid input shape");
    if (output.shape()[0] != batch_size || 
        output.shape()[1] != sequence_length || 
        output.shape()[2] != hidden) throw std::invalid_argument("Invalid output shape");
    
    for (std::size_t b = 0; b < batch_size; ++b) {
        for (std::size_t s = 0; s < sequence_length; ++s) {
            // Compute RMS
            float sum = 0.0f;
            for (std::size_t h = 0; h < hidden; ++h) {
                float curr = input[b * sequence_length * hidden + s * hidden + h];
                sum += curr * curr;
            }
            float rms = std::sqrt(sum / hidden + epsilon);

            // Populate output
            for (std::size_t h = 0; h < hidden; ++h) {
                output[b * sequence_length * hidden + s * hidden + h] = 
                    input[b * sequence_length * hidden + s * hidden + h] / rms * weights[h];
            }
        }
    }
}

} // namespace llmetal::cpu
