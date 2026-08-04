#pragma once

#include <cstddef>
#include <span>

#include "llmetal/gemv_shape.hpp"

namespace llmetal::cpu {

void gemv_f32(
    GemvShape shape,
    std::span<const float> matrix,
    std::span<const float> vector,
    std::span<float> output
);

} // namespace llmetal::cpu
