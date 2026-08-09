#include "llmetal/gemv_NRPSG.hpp"
#include "llmetal/gemv_mps.hpp"
#include "llmetal/tensor.hpp"
#include <llmetal/metal_context.hpp>
#include <llmetal/gemv_naive.hpp>

#include <cmath>
#include <cstddef>
#include <exception>
#include <algorithm>
#include <iostream>
#include <random>
#include <vector>

#include <Foundation/Foundation.h>
#include <Metal/Metal.h>
#include <MetalPerformanceShaders/MetalPerformanceShaders.h>

namespace {

template <typename Kernel>
bool run_case(llmetal::MetalContext& context, Kernel& kernel, llmetal::Shape shape) {
    std::size_t cols = shape[1];
    std::size_t rows = shape[0];
    std::size_t element_count = cols * rows;
    std::vector<float> matrix(element_count);
    std::vector<float> vector(cols);
    
    std::mt19937 rng(element_count);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    std::generate(matrix.begin(), matrix.end(), [&](){ return dist(rng); });
    std::generate(vector.begin(), vector.end(), [&](){ return dist(rng); });

    llmetal::CpuTensor<float> matrix_cpu({rows, cols}, matrix);
    llmetal::CpuTensor<float> vector_cpu({cols}, vector);
    llmetal::GpuTensor<float> matrix_gpu = context.upload(matrix_cpu);
    llmetal::GpuTensor<float> vector_gpu = context.upload(vector_cpu);
    llmetal::GpuTensor<float> output_gpu = context.allocate<float>({rows});
    
    auto job = kernel.submit(matrix_gpu, vector_gpu, output_gpu);
    job.wait();
    llmetal::CpuTensor<float> output_cpu = context.download(output_gpu);

    // Compute expected vector
    std::vector<float> expected(rows);
    for (std::size_t row = 0; row < rows; ++row) {
        for (std::size_t col = 0; col < cols; ++col) {
            expected[row] += matrix[row * cols + col] * vector[col];
        }
    }

    constexpr float tolerance = 1.0e-6f;
    for (std::size_t index = 0; index < rows; ++index) {
        if (std::fabs(output_cpu[index] - expected[index]) > tolerance) {
            std::cerr << "Mismatch at index " << index
                      << ": expected " << expected[index]
                      << ", got " << output_cpu[index] << '\n';

            // Print out input matrix and vector to help debugging.
            std::cout << "Matrix:\n";
            for (std::size_t row = 0; row < rows; ++row) {
                for (std::size_t col = 0; col < cols; ++col) {
                    std::cout << matrix[row * cols + col] << " ";
                }
                std::cout << "\n";
            }
            std::cout << "Vector:\n";
            for (std::size_t col = 0; col < cols; ++col) {
                std::cout << vector[col] << " ";
            }
            std::cout << "\n" << "Output:\n";
            for (std::size_t col = 0; col < cols; ++col) {
                std::cout << output_cpu[col] << " ";
            }
            std::cout << "\n" << "Expected:\n";
            for (std::size_t row = 0; row < rows; ++row) {
                std::cout << expected[row] << " ";
            }
            std::cout << std::endl;


            return false;
        }
    }

    return true;
}

// TODO: Unify the two somehow, and make them share the common code.
// TODO: make metal context able to allocate the MPSMatrix and MPSVector instead.
bool run_case(llmetal::MetalContext& context, llmetal::GemvMpsKernel& kernel, llmetal::Shape shape) {
    // Prepare buffers and sizes
    std::size_t cols = shape[1];
    std::size_t rows = shape[0];
    std::size_t element_count = cols * rows;
    std::vector<float> matrix(element_count);
    std::vector<float> vector(cols);
    std::vector<float> output(rows);
    const NSUInteger rowBytes = cols * sizeof(float);
    const NSUInteger colBytes = rows * sizeof(float);
    const NSUInteger matrixBytes = rows * rowBytes;
    
    // Generate the data
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
    std::vector<float> expected(rows);
    for (std::size_t row = 0; row < rows; ++row) {
        for (std::size_t col = 0; col < cols; ++col) {
            expected[row] += matrix[row * cols + col] * vector[col];
        }
    }

    constexpr float tolerance = 1.0e-6f;
    for (std::size_t index = 0; index < rows; ++index) {
        if (std::fabs(output[index] - expected[index]) > tolerance) {
            std::cerr << "Mismatch at index " << index
                      << ": expected " << expected[index]
                      << ", got " << output[index] << '\n';

            // Print out input matrix and vector to help debugging.
            std::cout << "Matrix:\n";
            for (std::size_t row = 0; row < rows; ++row) {
                for (std::size_t col = 0; col < cols; ++col) {
                    std::cout << matrix[row * cols + col] << " ";
                }
                std::cout << "\n";
            }
            std::cout << "Vector:\n";
            for (std::size_t col = 0; col < cols; ++col) {
                std::cout << vector[col] << " ";
            }
            std::cout << "\n" << "Output:\n";
            for (std::size_t col = 0; col < cols; ++col) {
                std::cout << output[col] << " ";
            }
            std::cout << "\n" << "Expected:\n";
            for (std::size_t row = 0; row < rows; ++row) {
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
        llmetal::GemvNRPSGKernel nrpsg(context, 1, 1);
        llmetal::GemvMpsKernel mps(context);

        // Exercises small work, a larger dispatch, and reuse after shrinking.
        for (const auto shape: {
            llmetal::Shape{4, 4},
            llmetal::Shape{16, 64},
            llmetal::Shape{32, 32}, // Same capacity, different dimensions
            llmetal::Shape{64, 32},

        }) {
            if (!run_case(context, naive, shape) || 
                !run_case(context, nrpsg, shape) ||
                !run_case(context, mps, shape)
            ) 
                return 1;
        }
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "Vector-add smoke test failed: " << error.what() << '\n';
        return 1;
    }
}
