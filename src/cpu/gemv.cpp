

#include "llmetal/gemv_shape.hpp"

#include <span>

namespace llmetal::cpu {

void gemv_f32(
    GemvShape shape,
    std::span<const float> matrix,
    std::span<const float> vector,
    std::span<float> output
) {
    if (matrix.size() != shape.rows * shape.cols ||
        vector.size() != shape.cols ||
        output.size() != shape.rows) {
        throw std::invalid_argument("Invalid shape or span size");
    }

    for (std::size_t row = 0; row < shape.rows; ++row) {
        float sum = 0.0f;
        for (std::size_t col = 0; col < shape.cols; ++col) {
            sum += matrix[row * shape.cols + col] * vector[col];
        }
        output[row] = sum;
    }
}

} // namespace llmetal::cpu
