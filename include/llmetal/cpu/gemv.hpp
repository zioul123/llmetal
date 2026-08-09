#pragma once

#include <cstddef>
#include <span>

#include "llmetal/tensor.hpp"

namespace llmetal::cpu {

// TODO: Fix clangd type hint for directories nested within include/llmetal

void gemv_f32(
    Shape shape,
    std::span<const float> matrix,
    std::span<const float> vector,
    std::span<float> output
);

} // namespace llmetal::cpu
