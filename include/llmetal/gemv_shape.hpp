#pragma once

#include <cstddef>

namespace llmetal {

struct GemvShape {
    std::size_t rows; // Output elements
    std::size_t cols; // Input elements
};

} // namespace llmetal
