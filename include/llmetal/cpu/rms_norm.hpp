#pragma once

#include "llmetal/tensor.hpp"

namespace llmetal::cpu {

void rms_norm(
    const llmetal::CpuTensor<float>& input,   // [batch_size, sequence_length, hidden]
    const llmetal::CpuTensor<float>& weights, // [hidden]
    llmetal::CpuTensor<float>& output,        // [batch_size, sequence_length, hidden]
    const float epsilon
);

} // namespace llmetal::cpu
