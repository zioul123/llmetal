#pragma once

#include "llmetal/tensor.hpp"

namespace llmetal::cpu {

void gqaPrefill(
    const CpuTensor<float>& q, // [B, S, NQ (num query heads), D (hidden dim per head)]
    const CpuTensor<float>& k, // [B, S, NKV, D]
    const CpuTensor<float>& v, // [B, S, NKV, D]
    CpuTensor<float>& output   // [B, S, NQ, D]
); 

void gqaPrefillFlash(
    const CpuTensor<float>& q, // [B, S, NQ (num query heads), D (hidden dim per head)]
    const CpuTensor<float>& k, // [B, S, NKV, D]
    const CpuTensor<float>& v, // [B, S, NKV, D]
    CpuTensor<float>& output,  // [B, S, NQ, D]
    const uint BQ = 2,
    const uint BK = 2
);

} // namespace llmetal::cpu
