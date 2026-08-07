#include <cstddef>

#include <llmetal/tensor.hpp>

namespace llmetal::cpu {

void embedding(
    llmetal::CpuTensor<float> &table,   // [vocab_size, hidden_size]
    llmetal::CpuTensor<std::uint32_t> &ids, // [batch_size, sequence_length]
    llmetal::CpuTensor<float> &output   // [batch_size, sequence_length, hidden_size]
) {
    if (table.shape().rank() != 2) {
        throw std::invalid_argument("Invalid table shape");
    }
    if (ids.shape().rank() != 2) {
        throw std::invalid_argument("Invalid ids shape");
    }
    std::size_t vocab_size = table.shape()[0];
    std::size_t hidden_size = table.shape()[1];
    std::size_t batch_size = ids.shape()[0];
    std::size_t sequence_length = ids.shape()[1];
    if (output.shape().rank() != 3) {
        throw std::invalid_argument("Invalid output shape");
    }
    if (output.shape()[0 ] != batch_size || 
        output.shape()[1] != sequence_length ||
        output.shape()[2] != hidden_size) {
        throw std::invalid_argument("Invalid output shape");
    }

    for (std::size_t b = 0; b < batch_size; ++b) {
        for (std::size_t s = 0; s < sequence_length; ++s) {
            for (std::size_t h = 0; h < hidden_size; ++h) {
                long long token_id = ids[b * sequence_length + s];
                output[b * sequence_length * hidden_size + s * hidden_size + h] = table[token_id * hidden_size + h];
            }
        }
    }
}

} // namespace llmetal::cpu
