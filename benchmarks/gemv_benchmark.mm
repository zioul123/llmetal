#include "llmetal/gemv_mps.hpp"
#include "llmetal/gemv_shape.hpp"
#include "llmetal/metal_context.hpp"
#include "llmetal/gemv_naive.hpp"
#include "llmetal/gemv_interleaved.hpp"
#include "llmetal/cpu/gemv.hpp"

#include <cmath>
#include <span>
#include <sstream>
#include <string>
#include <vector>
#include <chrono>
#include <concepts>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <random>
#include <string_view>
#include <iomanip>

// Aliases for brevity
using us = std::chrono::microseconds;
using ns = std::chrono::nanoseconds;
auto now = std::chrono::steady_clock::now;

template <typename K>
concept GemvKernel = requires(
    K& kernel,
    llmetal::GemvShape shape,
    std::span<const float> matrix,
    std::span<const float> vector,
    std::span<float> output,
    std::size_t repeats
) {
    { kernel.prepare(shape) } -> std::same_as<void>;
    { kernel.upload_matrix(matrix) } -> std::same_as<void>;
    { kernel.upload_vector(vector) } -> std::same_as<void>;
    { kernel.submit() } -> std::same_as<llmetal::MetalJob>;
    { kernel.submit_repeated(repeats) } -> std::same_as<llmetal::MetalJob>;
    { kernel.download(output) } -> std::same_as<void>;
};

struct BenchmarkDetails {
    std::string_view name;
    llmetal::GemvShape shape;
};

struct BenchmarkConfig {
    std::string_view name;
    llmetal::GemvShape shape;
    std::span<const float> matrix;
    std::span<const float> vector;

    std::size_t warmup_runs = 20;
    std::size_t timed_runs = 100;
};

constexpr int backend_width = 12;
constexpr int shape_width   = 12;
constexpr int time_width    = 16;
constexpr int total_time_width = 19;  // 12 digits + " us"
struct BenchmarkResult {
    std::string backend;
    BenchmarkConfig config;

    std::chrono::nanoseconds single_host_min;
    std::chrono::nanoseconds single_host_median;
    std::chrono::nanoseconds single_host_p95;
    std::chrono::nanoseconds batched_host_per_dispatch;
    std::chrono::nanoseconds batched_gpu_per_dispatch;

    void printHeader() const {
        std::cout
            << "Benchmark:\n"
            << "  "
            << std::left
            << std::setw(backend_width) << "Backend"
            << std::setw(shape_width) << "Matrix Size"
            << std::right
            << std::setw(total_time_width) << "Batch GPU/op"
            << std::setw(total_time_width) << "Batch Host/op"
            << std::setw(total_time_width) << "Submit + Wait Min"
            << std::setw(total_time_width) << "Submit + Wait Med"
            << std::setw(total_time_width) << "Submit + Wait P95"
            << '\n';
    }
    
    void printRow() const {
        std::cout
            << "  "
            << std::left
            << std::setw(backend_width) << backend
            << std::setw(shape_width)
            << (std::to_string(config.shape.rows) + "x" +
                std::to_string(config.shape.cols))
            << std::right
            << std::setw(time_width)
            << std::chrono::duration_cast<us>(batched_gpu_per_dispatch).count() << " us"
            << std::setw(time_width)
            << std::chrono::duration_cast<us>(batched_host_per_dispatch).count() << " us"
            << std::setw(time_width)
            << std::chrono::duration_cast<us>(single_host_min).count() << " us"
            << std::setw(time_width)
            << std::chrono::duration_cast<us>(single_host_median).count() << " us"
            << std::setw(time_width)
            << std::chrono::duration_cast<us>(single_host_p95).count() << " us"
            << '\n';
    }
};

struct ValidationFixture {
    std::string_view name;
    llmetal::GemvShape shape;
    std::span<const float> matrix;
    std::span<const float> vector;
    std::span<const float> expect;
};

struct ValidationResult {
    std::string backend;
    llmetal::GemvShape shape;

    bool valid;
    std::string error;
};

namespace Benchmark {

template <GemvKernel Kernel>
BenchmarkResult run(Kernel& kernel, BenchmarkConfig config, std::string_view backend) {
    if (config.timed_runs == 0) throw std::invalid_argument("timed_runs must be greater than 0");

    // Prepare kernel
    std::vector<float> output(config.shape.rows);
    kernel.prepare(config.shape);
    kernel.upload_matrix(config.matrix);
    kernel.upload_vector(config.vector);
    
    // Run warmup rounds
    if (config.warmup_runs != 0) {
        auto job = kernel.submit_repeated(config.warmup_runs);
        job.wait();
        kernel.download(output);
    }

    // Run timed rounds for single host time
    std::vector<std::chrono::nanoseconds> e2eDurations;
    {
        for (std::size_t i = 0; i < config.timed_runs; ++i) {
            auto start = now();
            auto job = kernel.submit();
            job.wait();
            auto stop = now();
            auto duration = stop - start;
            e2eDurations.push_back(
                std::chrono::duration_cast<ns>(duration)
            );
            start = stop;
        }
    }
    std::sort(e2eDurations.begin(), e2eDurations.end());
    const auto host_min = e2eDurations[0];
    const auto host_median = e2eDurations[config.timed_runs / 2];
    const auto p95_index = (config.timed_runs * 95 + 99) / 100 - 1;
    const auto host_p95 = e2eDurations[p95_index];

    // Run timed batched rounds
    ns batchGpuDuration;
    ns batchHostDuration;
    {
        auto start = now();
        auto job = kernel.submit_repeated(config.timed_runs);
        job.wait();
        auto stop = now();
    
        batchHostDuration = (stop - start) / config.timed_runs;
        batchGpuDuration = job.gpu_duration() / config.timed_runs;
    }

    return {
        .backend = std::string(backend),
        .config = config,
        .single_host_min = host_min,
        .single_host_median = host_median,
        .single_host_p95 = host_p95,
        .batched_host_per_dispatch = batchHostDuration,
        .batched_gpu_per_dispatch = batchGpuDuration,
    };
}

} // namespace Benchmark

namespace Validate {

ValidationResult validate_output(
    std::string_view backend,
    const llmetal::GemvShape& shape,
    std::span<const float> actual,
    std::span<const float> expect
) {
    // constexpr float absoluteTolerance = 1.0e-4f;
    constexpr float absoluteTolerance = 1.0e-3f;
    constexpr float relativeTolerance = 1.0e-4f;
    // constexpr float relativeTolerance = 1.0e-5f;
    for (std::size_t i = 0; i < actual.size(); ++i) {
        const float difference = std::fabs(actual[i] - expect[i]);
        const float scale = std::max(std::fabs(actual[i]), std::fabs(expect[i]));
        if (difference > absoluteTolerance + relativeTolerance * scale) {
            std::ostringstream message;
            message
                << "output[" << i << "] = " << actual[i]
                << ", expected " << expect[i];            
            return {
                .backend = std::string(backend),
                .shape = shape,
                .valid = false,
                .error = message.str(),
            };
        }
    }

    return {
        .backend = std::string(backend),
        .shape = shape,
        .valid = true,
        .error = {},
    };
}

template <GemvKernel Kernel>
ValidationResult run(Kernel& kernel, const ValidationFixture& fixture, std::string_view backend) {
    // Run the kernel
    std::vector<float> output(fixture.shape.rows);
    kernel.prepare(fixture.shape);
    kernel.upload_matrix(fixture.matrix);
    kernel.upload_vector(fixture.vector);
    auto job = kernel.submit();
    job.wait();
    kernel.download(output);

    // Validate
    return validate_output(
        std::string(backend), 
        fixture.shape, 
        output, 
        fixture.expect
    );
}

} // namespace Validate

int main() {
    try {
        llmetal::MetalContext context;
        llmetal::GemvNaiveKernel naiveKernel(context);
        llmetal::GemvMpsKernel mpsKernel(context);
        llmetal::GemvInterleavedKernel interleavedKernel(context);
        
        BenchmarkDetails details[] = {
            { "1024x1024", 1024, 1024},
            { "2048x1024", 2048, 1024},
            { "1024x2048", 1024, 2048},
            { "3072x1024", 3072, 1024},
            { "1024x3072", 1024, 3072},
            { "4096x4096", 4096, 4096},
            { "8192x8192", 8192, 8192},
            { "11008x4096", 11008, 4096},
            { "4096x11008", 4096, 11008},
            
        };

        for (auto const &detail : details) {
            std::cout << "=== " << detail.name << " ===" << std::endl;

            // === Preparation ===

            // Generate random data
            std::size_t element_count = detail.shape.cols * detail.shape.rows;
            std::vector<float> matrix(element_count);
            std::vector<float> vector(detail.shape.cols);
            std::vector<float> expect(detail.shape.rows);
            BenchmarkConfig config = {
                detail.name, detail.shape, matrix, vector
            };
            
            std::mt19937 rng(element_count);
            std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
            std::generate(matrix.begin(), matrix.end(), [&](){ return dist(rng); });
            std::generate(vector.begin(), vector.end(), [&](){ return dist(rng); });

            // === Validation pass ===

            // Produce CPU result and fixture
            llmetal::cpu::gemv_f32(config.shape, matrix, vector, expect);
            ValidationFixture fixture {
                config.name, config.shape, matrix, vector, expect 
            };

            // Validate the implementations
            auto naiveValidation = Validate::run(naiveKernel, fixture, "naive");
            auto mpsValidation = Validate::run(mpsKernel, fixture, "mps");
            auto interleavedValidation = Validate::run(interleavedKernel, fixture, "interleaved");
            // ValidationResult transposedValidation = Validate::transposed(context, config);
            // ValidationResult interleavedValidation = Validate::interleaved(context, config);

            // Print validation results
            std::cout << "Validation:\n";
            for (auto const &result : {
                naiveValidation, mpsValidation, interleavedValidation // transposedValidation, interleavedValidation
            }) {
                if (!result.valid) {
                    std::cerr << "  " << result.backend << ": failed - " << result.error << '\n';
                } else {
                    std::cout << "  " << result.backend << ": passed\n";
                }
            }
            if (!naiveValidation.valid || !mpsValidation.valid || !interleavedValidation.valid) {
                std::cerr << "Skipping benchmarks because validation failed." << std::endl;
                return 1;
            }

            // === Benchmark pass ===
            
            auto naiveBenchmark = Benchmark::run(naiveKernel, config, "naive");
            auto mpsBenchmark = Benchmark::run(mpsKernel, config, "mps");
            auto interleavedBenchmark = Benchmark::run(interleavedKernel, config, "interleaved");
            // BenchmarkResult transposedBenchmark = Benchmark::transposed(context, config);
            std::cout << std::endl;

            // Table header
            naiveBenchmark.printHeader();

            for (auto const &result : {
                naiveBenchmark, mpsBenchmark, interleavedBenchmark, // transposedBenchmark
            }) {
                result.printRow();
            }
            
            std::cout << std::endl;
        }
        
    } catch (const std::exception& error) {
        std::cerr << "GEMV Benchmark failed: " << error.what() << '\n';
        return 1;
    }
}
