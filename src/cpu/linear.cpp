#include "llmetal/cpu/linear.hpp"

namespace llmetal::cpu {

void linear(
    const llmetal::CpuTensor<float>& x,      // [B, S, I]
    const llmetal::CpuTensor<float>& weight, // [O, I]
    llmetal::CpuTensor<float>& y             // [B, S, O]
) {
    if (x.shape().rank() != 3) throw std::invalid_argument("Invalid input shape rank");
    if (weight.shape().rank() != 2) throw std::invalid_argument("Invalid weights shape rank");
    if (y.shape().rank() != 3) throw std::invalid_argument("Invalid output shape rank");

    std::uint32_t B = checked_u32(x.shape()[0], "cpu_linear_batch_size");
    std::uint32_t S = checked_u32(x.shape()[1], "cpu_linear_sequence_length");
    std::uint32_t O = checked_u32(weight.shape()[0], "cpu_rms_norm_hidden_out");
    std::uint32_t I = checked_u32(weight.shape()[1], "cpu_rms_norm_hidden_in");

    if (x.shape()[2] != I) throw std::invalid_argument("Invalid input shape");
    if (y.shape()[0] != B || 
        y.shape()[1] != S || 
        y.shape()[2] != O) throw std::invalid_argument("Invalid output shape");
    
    for (std::size_t b = 0; b < B; ++b) {
        for (std::size_t s = 0; s < S; ++s) {
            std::size_t bs_in_offset = (b * S + s) * I;
            std::size_t bs_out_offset = (b * S + s) * O;
            
            for (std::size_t o = 0; o < O; ++o) { 
                float weight_offset = o * I;
                float curr = 0.0f;
                for (std::size_t i = 0; i < I; ++i) {
                    curr += weight[weight_offset + i] * x[bs_in_offset + i];
                }
                y[bs_out_offset + o] = curr;
            }
        }
    }
}

void linear(
    const llmetal::CpuTensor<float>& x,      // [B, S, I]
    const llmetal::CpuTensor<float>& weight, // [O, I]
    const llmetal::CpuTensor<float>& bias,   // [O]
    llmetal::CpuTensor<float>& y             // [B, S, O]
) {
    if (x.shape().rank() != 3) throw std::invalid_argument("Invalid input shape rank");
    if (weight.shape().rank() != 2) throw std::invalid_argument("Invalid weights shape rank");
    if (y.shape().rank() != 3) throw std::invalid_argument("Invalid output shape rank");
    if (bias.shape().rank() != 1) throw std::invalid_argument("Invalid bias shape rank");

    std::uint32_t B = checked_u32(x.shape()[0], "cpu_linear_batch_size");
    std::uint32_t S = checked_u32(x.shape()[1], "cpu_linear_sequence_length");
    std::uint32_t O = checked_u32(weight.shape()[0], "cpu_rms_norm_hidden_out");
    std::uint32_t I = checked_u32(weight.shape()[1], "cpu_rms_norm_hidden_in");

    if (x.shape()[2] != I) throw std::invalid_argument("Invalid input shape");
    if (y.shape()[0] != B || 
        y.shape()[1] != S || 
        y.shape()[2] != O) throw std::invalid_argument("Invalid output shape");
    if (bias.shape()[0] != O) throw std::invalid_argument("Invalid bias shape");
    
    for (std::size_t b = 0; b < B; ++b) {
        for (std::size_t s = 0; s < S; ++s) {
            std::size_t bs_in_offset = (b * S + s) * I;
            std::size_t bs_out_offset = (b * S + s) * O;
            
            for (std::size_t o = 0; o < O; ++o) { 
                float weight_offset = o * I;
                float curr = 0.0f;
                for (std::size_t i = 0; i < I; ++i) {
                    curr += weight[weight_offset + i] * x[bs_in_offset + i];
                }
                y[bs_out_offset + o] = curr + bias[o];
            }
        }
    }
}

} // namespace llmetal::cpu
