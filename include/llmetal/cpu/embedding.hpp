#pragma once

#include "llmetal/tensor.hpp"
#include <cstdint>

namespace llmetal::cpu {

void embedding(
    llmetal::CpuTensor<float> &table,   // [vocab_size, hidden_size]
    llmetal::CpuTensor<std::uint32_t> &ids, // [batch_size, sequence_length]
    llmetal::CpuTensor<float> &output   // [batch_size, sequence_length, hidden_size]
);

} // namespace llmetal::cpu
