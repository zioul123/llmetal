#include "llmetal/gemv_shape.hpp"
#include "llmetal/metal_context.hpp"
#include "llmetal/gemv_naive.hpp"
#include "llmetal/cpu/gemv.hpp"

#include <chrono>
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

struct BenchmarkConfig {
    std::string_view name;
    llmetal::GemvShape shape;
    std::span<float> matrix;
    std::span<float> vector;

    std::size_t warmup_runs = 20;
    std::size_t timed_runs = 1000;
};

constexpr int backend_width = 12;
constexpr int shape_width   = 12;
constexpr int time_width    = 12;
constexpr int total_time_width = 15;  // 12 digits + " us"
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
            << std::setw(total_time_width) << "E2E Host Min"
            << std::setw(total_time_width) << "E2E Host Med"
            << std::setw(total_time_width) << "E2E Host P95"
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
    std::span<float> matrix;
    std::span<float> vector;
    std::span<float> expect;
};

struct ValidationResult {
    std::string backend;
    llmetal::GemvShape shape;

    bool valid;
    std::string error;
};

namespace Benchmark {

BenchmarkResult naive(llmetal::MetalContext& context, BenchmarkConfig config) {
    // Prepare kernel
    llmetal::GemvNaiveKernel kernel(context);
    std::vector<float> output(config.shape.rows);
    kernel.prepare(config.shape);
    kernel.upload_matrix(config.matrix);
    kernel.upload_vector(config.vector);
    
    // Run warmup rounds
    {
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
        .backend = "naive",
        .config = config,
        .single_host_min = host_min,
        .single_host_median = host_median,
        .single_host_p95 = host_p95,
        .batched_host_per_dispatch = batchHostDuration,
        .batched_gpu_per_dispatch = batchGpuDuration,
    };
}

BenchmarkResult mps(llmetal::MetalContext& context, BenchmarkConfig config) {
    llmetal::GemvNaiveKernel kernel(context);
    auto start = now();
    auto stop = now();
    auto duration = stop - start;

    return {
        .backend = "mps",
        .config = config,
        .single_host_min = duration,
        .single_host_median = duration,
        .single_host_p95 = duration,
        .batched_host_per_dispatch = duration,
        .batched_gpu_per_dispatch = duration
    };
}


BenchmarkResult transposed(llmetal::MetalContext& context, BenchmarkConfig config) {
    llmetal::GemvNaiveKernel kernel(context);
    auto start = now();
    auto stop = now();
    auto duration = stop - start;

    return {
        .backend = "transposed",
        .config = config,
        .single_host_min = duration,
        .single_host_median = duration,
        .single_host_p95 = duration,
        .batched_host_per_dispatch = duration,
        .batched_gpu_per_dispatch = duration
    };
}

BenchmarkResult interleaved(llmetal::MetalContext& context, BenchmarkConfig config) {
    llmetal::GemvNaiveKernel kernel(context);
    auto start = now();
    auto stop = now();
    auto duration = stop - start;

    return {
        .backend = "interleaved",
        .config = config,
        .single_host_min = duration,
        .single_host_median = duration,
        .single_host_p95 = duration,
        .batched_host_per_dispatch = duration,
        .batched_gpu_per_dispatch = duration
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
    constexpr float tolerance = 1.0e-6f;

    for (std::size_t i = 0; i < actual.size(); ++i) {
        if (std::fabs(actual[i] - expect[i]) > tolerance) {
            std::ostringstream message;
            message
                << "output[" << i << "] = " << actual[i]
                << ", expected " << expect[i];

            std::cerr << "    " << backend << ": " << message.str() << '\n';
            
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

ValidationResult naive(llmetal::MetalContext& context, const ValidationFixture& fixture) {
    std::vector<float> output(fixture.shape.rows);

    // Run the kernel
    llmetal::GemvNaiveKernel kernel(context);
    kernel.prepare(fixture.shape);
    kernel.upload_matrix(fixture.matrix);
    kernel.upload_vector(fixture.vector);
    auto job = kernel.submit();
    job.wait();
    kernel.download(output);

    // Validate
    return validate_output(
        "naive", 
        fixture.shape, 
        output, 
        fixture.expect
    );
}

ValidationResult mps(llmetal::MetalContext& context, const ValidationFixture& fixture) {
    // llmetal::GemvKernel kernel(context);
    

    return {
        .backend = "mps",
        .shape = fixture.shape,
        .valid = true,
        .error = "",
    };
}


ValidationResult transposed(llmetal::MetalContext& context, const ValidationFixture& fixture) {
    llmetal::GemvNaiveKernel kernel(context);
    auto start = std::chrono::high_resolution_clock::now();
    auto stop = std::chrono::high_resolution_clock::now();
    auto duration = stop - start;

    return {
        .backend = "transposed",
        .shape = fixture.shape,
        .valid = true,
        .error = "",
    };
}

ValidationResult interleaved(llmetal::MetalContext& context, const ValidationFixture& fixture) {
    llmetal::GemvNaiveKernel kernel(context);
    auto start = std::chrono::high_resolution_clock::now();
    auto stop = std::chrono::high_resolution_clock::now();
    auto duration = stop - start;

    return {
        .backend = "interleaved",
        .shape = fixture.shape,
        .valid = true,
        .error = "",
    };
}

} // namespace Validate

int main() {
    try {
        llmetal::MetalContext context;
        BenchmarkConfig configs[] = {
            { "1024x1024", 1024, 1024},
            { "2048x1024", 2048, 1024},
            { "1024x2048", 1024, 2048},
            { "3072x1024", 3072, 1024},
            { "1024x3072", 1024, 3072}
        };

        for (auto const &config : configs) {
            std::cout << "=== " << config.name << " ===" << std::endl;

            // Generate random data
            std::size_t element_count = config.shape.cols * config.shape.rows;
            std::vector<float> matrix(element_count);
            std::vector<float> vector(config.shape.cols);
            std::vector<float> expect(config.shape.rows);
            
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
            ValidationResult naiveValidation = Validate::naive(context, fixture);
            // ValidationResult mpsValidation = Validate::mps(context, fixture);
            // ValidationResult transposedValidation = Validate::transposed(context, config);
            // ValidationResult interleavedValidation = Validate::interleaved(context, config);

            // Print validation results
            std::cout << "Validation:\n";
            for (auto const &result : {
                naiveValidation, // mpsValidation, transposedValidation, interleavedValidation
            }) {
                if (!result.valid) {
                    std::cerr << "  " << result.backend << ": " << result.error << '\n';
                } else {
                    std::cout << "  " << result.backend << ": passed\n";
                }
            }

            // === Benchmark pass ===

            BenchmarkResult naiveBenchmark = Benchmark::naive(
                context, 
                {
                    .name=config.name,
                    .shape=config.shape,
                    .matrix=matrix,
                    .vector=vector
                }
            );
            // BenchmarkResult mpsBenchmark = Benchmark::mps(context, config);
            // BenchmarkResult transposedBenchmark = Benchmark::transposed(context, config);
            // BenchmarkResult interleavedBenchmark = Benchmark::interleaved(context, config);
            
            // Table header
            naiveBenchmark.printHeader();

            for (auto const &result : {
                naiveBenchmark, // mpsBenchmark, transposedBenchmark, interleavedBenchmark
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
