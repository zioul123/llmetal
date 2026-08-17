#pragma once

#include "llmetal/tensor.hpp"

namespace llmetal::cpu {

void linear(
    const llmetal::CpuTensor<float>& x,      // [B, S, I]
    const llmetal::CpuTensor<float>& weight, // [O, I]
    llmetal::CpuTensor<float>& y             // [B, S, O]
);

void linear(
    const llmetal::CpuTensor<float>& x,      // [B, S, I]
    const llmetal::CpuTensor<float>& weight, // [O, I]
    const llmetal::CpuTensor<float>& bias,   // [O]
    llmetal::CpuTensor<float>& y             // [B, S, O]
);

} // namespace llmetal::cpu
