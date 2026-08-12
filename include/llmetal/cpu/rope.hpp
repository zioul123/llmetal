#pragma once

#include "llmetal/tensor.hpp"

namespace llmetal::cpu {

void rope_cos_and_sin(
    const std::uint32_t max_seq_length,
    const std::uint32_t rotary_dim,
    const float theta,
    llmetal::CpuTensor<float>& output // [max_seq_length, rotary_dim / 2, 2 (cos_and_sin)]
);

} // namespace llmetal::cpu
