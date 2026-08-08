#include "llmetal/cpu/embedding.hpp"
#include "llmetal/embedding.hpp"
#include "llmetal/metal_context.hpp"
#include "llmetal/tensor.hpp"

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


struct BenchmarkConfig {
    std::string_view name;
    llmetal::Shape tableShape;
    llmetal::Shape idsShape;
};

struct BenchmarkConfigWithData {
    std::string_view name;
    llmetal::Shape tableShape;
    llmetal::Shape idsShape;
    const llmetal::GpuTensor<float>& table;
    const llmetal::GpuTensor<std::uint32_t>& ids;

    std::size_t warmup_runs = 20;
    std::size_t timed_runs = 100;
    std::size_t repeats = 30;
};

constexpr int backend_width = 14;
constexpr int shape_width   = 12;
constexpr int time_width    = 16;
constexpr int total_time_width = 19;  // 12 digits + " us"
struct BenchmarkResult {
    std::string backend;
    BenchmarkConfigWithData config;

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
            << std::setw(shape_width) << "Table Shape"
            << std::setw(shape_width) << "Ids Shape"
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
            << std::setw(shape_width) << config.tableShape.to_string()
            << std::setw(shape_width) << config.idsShape.to_string()
            << std::right
            << std::setw(time_width)
            << std::chrono::duration_cast<ns>(batched_gpu_per_dispatch).count() << " ns"
            << std::setw(time_width)
            << std::chrono::duration_cast<ns>(batched_host_per_dispatch).count() << " ns"
            << std::setw(time_width)
            << std::chrono::duration_cast<ns>(single_host_min).count() << " ns"
            << std::setw(time_width)
            << std::chrono::duration_cast<ns>(single_host_median).count() << " ns"
            << std::setw(time_width)
            << std::chrono::duration_cast<ns>(single_host_p95).count() << " ns"
            << '\n';
    }
};

struct ValidationFixture {
    std::string_view name;
    llmetal::Shape tableShape;
    llmetal::Shape idsShape;
    const llmetal::GpuTensor<float>& table;
    const llmetal::GpuTensor<std::uint32_t>& ids;
    const llmetal::CpuTensor<float>& expect;
};

struct ValidationResult {
    std::string backend;
    llmetal::Shape tableShape;
    llmetal::Shape idsShape;
    bool valid;
    std::string error;
};

namespace Benchmark {

BenchmarkResult run(
    llmetal::MetalContext& context, 
    llmetal::EmbeddingKernel& kernel, 
    const BenchmarkConfigWithData& config, 
    std::string_view backend
) {
    if (config.timed_runs == 0) throw std::invalid_argument("timed_runs must be greater than 0");
    std::uint32_t batch_size = checked_u32(config.idsShape[0], "batch_size");
    std::uint32_t sequence_length = checked_u32(config.idsShape[1], "sequence_length");
    std::uint32_t vocab_size = checked_u32(config.tableShape[0], "vocab_size");
    std::uint32_t hidden_size = checked_u32(config.tableShape[1], "hidden_size");

    // Prepare output tensor
    auto output_tensor_gpu = context.allocate<float>(
        llmetal::Shape{batch_size, sequence_length, hidden_size}
    );
    
    // Run warmup rounds
    if (config.warmup_runs != 0) {
        auto job = kernel.submit_repeated(
            config.warmup_runs, 
            config.table, 
            config.ids, 
            output_tensor_gpu
        );
        job.wait();
        auto result = context.download(output_tensor_gpu);
    }

    // Run timed rounds for single host time
    std::vector<std::chrono::nanoseconds> e2eDurations;
    {
        for (std::size_t i = 0; i < config.timed_runs; ++i) {
            auto start = now();
            auto job = kernel.submit(
                config.table, 
                config.ids, 
                output_tensor_gpu
            );
            job.wait();
            auto stop = now();
            auto duration = stop - start;
            e2eDurations.push_back(std::chrono::duration_cast<ns>(duration));
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
        auto job = kernel.submit_repeated(
            config.timed_runs, 
            config.table, 
            config.ids, 
            output_tensor_gpu
        );
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
    llmetal::Shape tableShape,
    llmetal::Shape idsShape,
    const llmetal::CpuTensor<float>& actual,
    const llmetal::CpuTensor<float>& expect
) {
    // constexpr float absoluteTolerance = 1.0e-4f;
    constexpr float absoluteTolerance = 1.0e-4f;
    constexpr float relativeTolerance = 1.0e-5f;
    if (expect.numel() != actual.numel()) {
        std::ostringstream message;
        message
            << "actual numel = " << actual.numel()
            << ", expect numel = " << expect.numel();
        return {
            .backend = std::string(backend),
            .tableShape = tableShape,
            .idsShape = idsShape,
            .valid = false,
            .error = message.str()
        };
    }
    // std::cout << "ACTUAL" << std::endl;
    // std::cout << actual << std::endl;
    // std::cout << "EXPECT" << std::endl;
    // std::cout << expect << std::endl;

    // constexpr float relativeTolerance = 1.0e-5f;
    for (std::size_t i = 0; i < actual.numel(); ++i) {
        const float difference = std::fabs(actual[i] - expect[i]);
        const float scale = std::max(std::fabs(actual[i]), std::fabs(expect[i]));
        if (difference > absoluteTolerance + relativeTolerance * scale) {
            std::ostringstream message;
            message
                << "output[" << i << "] = " << actual[i]
                << ", expected " << expect[i];            
            return {
                .backend = std::string(backend),
                .tableShape = tableShape,
                .idsShape = idsShape,
                .valid = false,
                .error = message.str(),
            };
        }
    }

    return {
        .backend = std::string(backend),
        .tableShape = tableShape,
        .idsShape = idsShape,
        .valid = true,
        .error = {},
    };
}

ValidationResult run(
    llmetal::MetalContext& context, 
    llmetal::EmbeddingKernel& kernel,
    const ValidationFixture& fixture,
    std::string_view backend
) {
    std::uint32_t batch_size = checked_u32(fixture.idsShape[0], "batch_size");
    std::uint32_t sequence_length = checked_u32(fixture.idsShape[1], "sequence_length");
    std::uint32_t vocab_size = checked_u32(fixture.tableShape[0], "vocab_size");
    std::uint32_t hidden_size = checked_u32(fixture.tableShape[1], "hidden_size");

    // Prepare output tensor
    auto output_tensor_gpu = context.allocate<float>(
        llmetal::Shape{batch_size, sequence_length, hidden_size}
    );

    // Run the kernel
    auto job = kernel.submit(
        fixture.table, 
        fixture.ids, 
        output_tensor_gpu
    );
    job.wait();
    auto output_tensor_cpu = context.download(output_tensor_gpu);

    // Validate
    return validate_output(
        std::string(backend), 
        fixture.tableShape,
        fixture.idsShape,
        output_tensor_cpu,
        fixture.expect
    );
}

} // namespace Validate

int main() {
    try {
        llmetal::MetalContext context;
        llmetal::EmbeddingKernel kernel_tg32_tpr1(context, 32, 1);
        llmetal::EmbeddingKernel kernel_tg32_tpr32(context, 32, 32);
        llmetal::EmbeddingKernel kernel_tg64_tpr64(context, 64, 64);
        llmetal::EmbeddingKernel kernel_tg128_tpr32(context, 128, 32);
        llmetal::EmbeddingKernel kernel_tg128_tpr64(context, 128, 64);
        llmetal::EmbeddingKernel kernel_tg128_tpr128(context, 128, 128);
        llmetal::EmbeddingKernel kernel_tg256_tpr32(context, 256, 32);
        llmetal::EmbeddingKernel kernel_tg256_tpr64(context, 256, 64);
        llmetal::EmbeddingKernel kernel_tg256_tpr128(context, 256, 128);
        llmetal::EmbeddingKernel kernel_tg256_tpr256(context, 256, 256);
        llmetal::EmbeddingKernel kernel_tg512_tpr32(context, 512, 32);
        llmetal::EmbeddingKernel kernel_tg512_tpr64(context, 512, 64);     // Best it seems
        llmetal::EmbeddingKernel kernel_tg512_tpr128(context, 512, 128);
        llmetal::EmbeddingKernel kernel_tg512_tpr256(context, 512, 256);
        llmetal::EmbeddingKernel kernel_tg512_tpr512(context, 512, 512);
        llmetal::EmbeddingKernel kernel_tg1024_tpr32(context, 1024, 32);
        llmetal::EmbeddingKernel kernel_tg1024_tpr64(context, 1024, 64);
        llmetal::EmbeddingKernel kernel_tg1024_tpr128(context, 1024, 128);
        llmetal::EmbeddingKernel kernel_tg1024_tpr256(context, 1024, 256);
        llmetal::EmbeddingKernel kernel_tg1024_tpr512(context, 1024, 512);
        
        BenchmarkConfig details[] = {
            BenchmarkConfig{
                .name="SmolLM2-3x10000",
                .tableShape={49152, 576},
                .idsShape={3, 10000}
            }
        };

        for (BenchmarkConfig const &config : details) {
            std::cout << "=== " << config.name << " ===" << std::endl;

            // === Preparation ===
            // Get all shapes
            std::uint32_t batch_size = checked_u32(config.idsShape[0], "batch_size");
            std::uint32_t sequence_length = checked_u32(config.idsShape[1], "sequence_length");
            std::uint32_t vocab_size = checked_u32(config.tableShape[0], "vocab_size");
            std::uint32_t hidden_size = checked_u32(config.tableShape[1], "hidden_size");

            llmetal::Shape table_shape{vocab_size, hidden_size};
            llmetal::Shape ids_shape{batch_size, sequence_length};
            llmetal::Shape expect_shape{batch_size, sequence_length, hidden_size};

            // Generate data
            std::vector<float> _table_cpu(vocab_size * hidden_size);
            std::vector<std::uint32_t> _ids_cpu(batch_size * sequence_length);
            std::mt19937 rng(batch_size * sequence_length * hidden_size * vocab_size);
            std::uniform_real_distribution<float> dist1(-1.0f, 1.0f);
            std::generate(_table_cpu.begin(), _table_cpu.end(), [&](){ return dist1(rng); });
            std::uniform_int_distribution<std::uint32_t> dist2(0, vocab_size - 1);
            std::generate(_ids_cpu.begin(), _ids_cpu.end(), [&](){ return dist2(rng) ; });

            // Create tensors and fixtures
            llmetal::CpuTensor<float> table_cpu({vocab_size, hidden_size}, _table_cpu);
            llmetal::CpuTensor<std::uint32_t> ids_cpu({batch_size, sequence_length}, _ids_cpu);
            llmetal::CpuTensor<float> expect_cpu({batch_size, sequence_length, hidden_size});

            llmetal::GpuTensor<float> table_gpu = context.upload<float>(table_cpu);
            llmetal::GpuTensor<std::uint32_t> ids_gpu = context.upload<std::uint32_t>(ids_cpu);
            
            // Validation config
            llmetal::cpu::embedding(table_cpu, ids_cpu, expect_cpu, vocab_size);
            ValidationFixture fixture {
                config.name,
                table_shape,
                ids_shape,
                table_gpu,
                ids_gpu,
                expect_cpu
            };
            // Benchmark config
            BenchmarkConfigWithData config_with_data = {
                config.name,
                table_shape,
                ids_shape,
                table_gpu,
                ids_gpu
            };

            // === Validation pass ===
            // Validate the implementations
            ValidationResult validationResults[] = {
                Validate::run(context, kernel_tg32_tpr1, fixture, "tg32_tpr1"),
                Validate::run(context, kernel_tg32_tpr32, fixture, "tg32_tpr32"),
                Validate::run(context, kernel_tg64_tpr64, fixture, "tg64_tpr64"),
                Validate::run(context, kernel_tg128_tpr32, fixture, "tg128_tpr32"),
                Validate::run(context, kernel_tg128_tpr64, fixture, "tg128_tpr64"),
                Validate::run(context, kernel_tg128_tpr128, fixture, "tg128_tpr128"),
                Validate::run(context, kernel_tg256_tpr32, fixture, "tg256_tpr32"),
                Validate::run(context, kernel_tg256_tpr64, fixture, "tg256_tpr64"),
                Validate::run(context, kernel_tg256_tpr128, fixture, "tg256_tpr128"),
                Validate::run(context, kernel_tg256_tpr256, fixture, "tg256_tpr256"),
                Validate::run(context, kernel_tg512_tpr32, fixture, "tg512_tpr32"),
                Validate::run(context, kernel_tg512_tpr64, fixture, "tg512_tpr64"),
                Validate::run(context, kernel_tg512_tpr128, fixture, "tg512_tpr128"),
                Validate::run(context, kernel_tg512_tpr256, fixture, "tg512_tpr256"),
                Validate::run(context, kernel_tg512_tpr512, fixture, "tg512_tpr512"),
                Validate::run(context, kernel_tg1024_tpr32, fixture, "tg1024_tpr32"),
                Validate::run(context, kernel_tg1024_tpr64, fixture, "tg1024_tpr64"),
                Validate::run(context, kernel_tg1024_tpr128, fixture, "tg1024_tpr128"),
                Validate::run(context, kernel_tg1024_tpr256, fixture, "tg1024_tpr256"),
                Validate::run(context, kernel_tg1024_tpr512, fixture, "tg1024_tpr512"),
            };
            
            // Print validation results
            std::cout << "Validation:\n";
            bool any_failed = false;
            for (auto const &result : validationResults) {
                if (!result.valid) {
                    std::cerr << "  " << result.backend << ": failed - " << result.error << '\n';
                    any_failed = true;
                }
            }
            if (any_failed) {
                std::cerr << "Skipping benchmarks because validation failed." << std::endl;
                return 1;
            } else {
                std::cout << "  Validation passed." << std::endl;
            }

            // === Benchmark pass ===
            BenchmarkResult benchmarkResults[] = {
                Benchmark::run(context, kernel_tg32_tpr1, config_with_data, "tg32_tpr1"),
                Benchmark::run(context, kernel_tg32_tpr32, config_with_data, "tg32_tpr32"),
                Benchmark::run(context, kernel_tg64_tpr64, config_with_data, "tg64_tpr64"),
                Benchmark::run(context, kernel_tg128_tpr32, config_with_data, "tg128_tpr32"),
                Benchmark::run(context, kernel_tg128_tpr64, config_with_data, "tg128_tpr64"),
                Benchmark::run(context, kernel_tg128_tpr128, config_with_data, "tg128_tpr128"),
                Benchmark::run(context, kernel_tg256_tpr32, config_with_data, "tg256_tpr32"),
                Benchmark::run(context, kernel_tg256_tpr64, config_with_data, "tg256_tpr64"),
                Benchmark::run(context, kernel_tg256_tpr128, config_with_data, "tg256_tpr128"),
                Benchmark::run(context, kernel_tg256_tpr256, config_with_data, "tg256_tpr256"),
                Benchmark::run(context, kernel_tg512_tpr32, config_with_data, "tg512_tpr32"),
                Benchmark::run(context, kernel_tg512_tpr64, config_with_data, "tg512_tpr64"),
                Benchmark::run(context, kernel_tg512_tpr128, config_with_data, "tg512_tpr128"),
                Benchmark::run(context, kernel_tg512_tpr256, config_with_data, "tg512_tpr256"),
                Benchmark::run(context, kernel_tg512_tpr512, config_with_data, "tg512_tpr512"),
                Benchmark::run(context, kernel_tg1024_tpr32, config_with_data, "tg1024_tpr32"),
                Benchmark::run(context, kernel_tg1024_tpr64, config_with_data, "tg1024_tpr64"),
                Benchmark::run(context, kernel_tg1024_tpr128, config_with_data, "tg1024_tpr128"),
                Benchmark::run(context, kernel_tg1024_tpr256, config_with_data, "tg1024_tpr256"),
                Benchmark::run(context, kernel_tg1024_tpr512, config_with_data, "tg1024_tpr512"),
            };
            // Table header
            benchmarkResults[0].printHeader();
            for (auto const &result : benchmarkResults) {
                result.printRow();
            }
            
            std::cout << std::endl;
        }
        
    } catch (const std::exception& error) {
        std::cerr << "GEMV Benchmark failed: " << error.what() << '\n';
        return 1;
    }
}
