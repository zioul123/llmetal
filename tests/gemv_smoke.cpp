#include "llmetal/gemv_mps.hpp"
#include "llmetal/gemv_shape.hpp"
#include <llmetal/metal_context.hpp>
#include <llmetal/gemv_naive.hpp>

#include <cmath>
#include <cstddef>
#include <exception>
#include <algorithm>
#include <iostream>
#include <random>
#include <vector>

namespace {

template <typename Kernel>
bool run_case(Kernel& kernel, llmetal::GemvShape shape) {
    std::size_t element_count = shape.cols * shape.rows;
    std::vector<float> matrix(element_count);
    std::vector<float> vector(shape.cols);
    std::vector<float> output(shape.rows);
    
    std::mt19937 rng(element_count);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    std::generate(matrix.begin(), matrix.end(), [&](){ return dist(rng); });
    std::generate(vector.begin(), vector.end(), [&](){ return dist(rng); });

    kernel.prepare(shape);
    kernel.upload_matrix(matrix);
    kernel.upload_vector(vector);
    auto job = kernel.submit();
    job.wait();
    kernel.download(output);

    // Compute expected vector
    std::vector<float> expected(shape.rows);
    for (std::size_t row = 0; row < shape.rows; ++row) {
        for (std::size_t col = 0; col < shape.cols; ++col) {
            expected[row] += matrix[row * shape.cols + col] * vector[col];
        }
    }

    constexpr float tolerance = 1.0e-6f;
    for (std::size_t index = 0; index < shape.rows; ++index) {
        if (std::fabs(output[index] - expected[index]) > tolerance) {
            std::cerr << "Mismatch at index " << index
                      << ": expected " << expected[index]
                      << ", got " << output[index] << '\n';

            // Print out input matrix and vector to help debugging.
            std::cout << "Matrix:\n";
            for (std::size_t row = 0; row < shape.rows; ++row) {
                for (std::size_t col = 0; col < shape.cols; ++col) {
                    std::cout << matrix[row * shape.cols + col] << " ";
                }
                std::cout << "\n";
            }
            std::cout << "Vector:\n";
            for (std::size_t col = 0; col < shape.cols; ++col) {
                std::cout << vector[col] << " ";
            }
            std::cout << "\n" << "Output:\n";
            for (std::size_t col = 0; col < shape.cols; ++col) {
                std::cout << output[col] << " ";
            }
            std::cout << "\n" << "Expected:\n";
            for (std::size_t row = 0; row < shape.rows; ++row) {
                std::cout << expected[row] << " ";
            }
            std::cout << std::endl;


            return false;
        }
    }

    return true;
}

} // namespace

int main() {
    try {
        llmetal::MetalContext context;
        llmetal::GemvNaiveKernel naive(context);
        llmetal::GemvMpsKernel mps(context);

        // Exercises small work, a larger dispatch, and reuse after shrinking.
        for (const auto shape: {
            llmetal::GemvShape{4, 4},
            llmetal::GemvShape{16, 64},
            llmetal::GemvShape{32, 32}, // Same capacity, different dimensions
            llmetal::GemvShape{64, 32},

        }) {
            if (!run_case(naive, shape) || !run_case(mps, shape)) return 1;
        }
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "Vector-add smoke test failed: " << error.what() << '\n';
        return 1;
    }
}
