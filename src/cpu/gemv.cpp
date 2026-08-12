#include "llmetal/cpu/gemv.hpp"

namespace llmetal::cpu {

void gemv_f32(
    Shape shape,
    std::span<const float> matrix,
    std::span<const float> vector,
    std::span<float> output
) {
    if (shape.rank() != 2) {
        throw std::invalid_argument("Invalid shape, expected 2, got " + std::to_string(shape.rank()));
    }
    std::size_t rows = shape[0];
    std::size_t cols = shape[1];

    if (matrix.size() != rows * cols ||
        vector.size() != cols ||
        output.size() != rows) {
        throw std::invalid_argument("Invalid shape or span size");
    }

    for (std::size_t row = 0; row < rows; ++row) {
        float sum = 0.0f;
        for (std::size_t col = 0; col < cols; ++col) {
            sum += matrix[row * cols + col] * vector[col];
        }
        output[row] = sum;
    }
}

} // namespace llmetal::cpu
