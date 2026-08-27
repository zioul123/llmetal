#pragma once

#include "llmetal/tensor.hpp"

namespace llmetal::cpu {

void softmax(
    const CpuTensor<float>& logits,        // [rows,C]
    const CpuTensor<std::uint32_t>& valid, // [rows]
    CpuTensor<float>& probs                // [rows,C]
);

} // namespace llmetal::cpu
