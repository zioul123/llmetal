#include "llmetal/cpu/softmax.hpp"
#include "llmetal/softmax.hpp"
#include "llmetal/metal_context.hpp"
#include "llmetal/tensor.hpp"

#include <cmath>
#include <cstdint>
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
    llmetal::Shape logitsShape;
    llmetal::Shape validShape;
};

struct BenchmarkConfigWithData {
    std::string_view name;
    llmetal::Shape logitsShape;
    llmetal::Shape validShape;
    const llmetal::GpuTensor<float>& logits;
    const llmetal::GpuTensor<std::uint32_t>& valid;

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
            << std::setw(shape_width) << "Input Shape"
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
            << std::setw(shape_width) << config.logitsShape.to_string()
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
    llmetal::Shape logitsShape;
    llmetal::Shape validShape;
    const llmetal::GpuTensor<float>& logits;
    const llmetal::GpuTensor<std::uint32_t>& valid;
    const llmetal::CpuTensor<float>& expect;
};

struct ValidationResult {
    std::string backend;
    llmetal::Shape inputsShape;
    llmetal::Shape weightsShape;
    bool valid;
    std::string error;
};

namespace Benchmark {

BenchmarkResult run(
    llmetal::MetalContext& context, 
    llmetal::SoftmaxKernel& kernel, 
    const BenchmarkConfigWithData& config, 
    std::string_view backend
) {
    if (config.timed_runs == 0) throw std::invalid_argument("timed_runs must be greater than 0");
    std::uint32_t rows = checked_u32(config.logitsShape[0], "batch_size");
    std::uint32_t sequence_length = checked_u32(config.logitsShape[1], "sequence_length");

    // Prepare output tensor
    auto output_tensor_gpu = context.allocate<float>({rows, sequence_length});
    
    // Run warmup rounds
    if (config.warmup_runs != 0) {
        auto job = kernel.submit_repeated(
            config.warmup_runs, 
            config.logits, 
            config.valid, 
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
                config.logits, 
                config.valid, 
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
            config.logits, 
            config.valid, 
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
    llmetal::Shape logitsShape,
    llmetal::Shape validShape,
    const llmetal::CpuTensor<float>& actual,
    const llmetal::CpuTensor<float>& expect
) {
    constexpr float absoluteTolerance = 1.0e-4f;
    constexpr float relativeTolerance = 1.0e-5f;
    
    if (expect.numel() != actual.numel()) {
        std::ostringstream message;
        message
            << "actual numel = " << actual.numel()
            << ", expect numel = " << expect.numel();
        return {
            std::string(backend),
            logitsShape,
            validShape,
            false,
            message.str()
        };
    }

    for (std::size_t i = 0; i < actual.numel(); ++i) {
        const float difference = std::fabs(actual[i] - expect[i]);
        const float scale = std::max(std::fabs(actual[i]), std::fabs(expect[i]));
        if (difference > absoluteTolerance + relativeTolerance * scale) {
            std::ostringstream message;
            message
                << "output[" << i << "] = " << actual[i]
                << ", expected " << expect[i];            
            return {
                std::string(backend),
                logitsShape,
                validShape,
                false,
                message.str(),
            };
        }
    }

    return {
        std::string(backend),
        logitsShape,
        validShape,
        true,
        {},
    };
}

ValidationResult run(
    llmetal::MetalContext& context, 
    llmetal::SoftmaxKernel& kernel,
    const ValidationFixture& fixture,
    std::string_view backend
) {
    std::uint32_t rows = checked_u32(fixture.logitsShape[0], "batch_size");
    std::uint32_t sequence_length = checked_u32(fixture.logitsShape[1], "sequence_length");
    // Prepare output tensor
    auto output_tensor_gpu = context.allocate<float>({rows, sequence_length});
    // Run the kernel
    auto job = kernel.submit(
        fixture.logits, 
        fixture.valid, 
        output_tensor_gpu
    );
    job.wait();
    auto output_tensor_cpu = context.download(output_tensor_gpu);

    // Validate
    return validate_output(
        std::string(backend), 
        fixture.logitsShape,
        fixture.validShape,
        output_tensor_cpu,
        fixture.expect
    );
}

} // namespace Validate

struct KernelAndBackend {
    llmetal::SoftmaxKernel kernel;
    std::string_view backend;
};

int main() {
    try {
        llmetal::MetalContext context;
        KernelAndBackend kernels[] = {
            KernelAndBackend{ llmetal::SoftmaxKernel(context, 32, 1, 1), "tg32_tpr1_rpt1" },
            // KernelAndBackend{ llmetal::SoftmaxKernel(context, 32, 32), "tg32_tpr32" },
            // KernelAndBackend{ llmetal::SoftmaxKernel(context, 64, 32), "tg64_tpr32" },
            // KernelAndBackend{ llmetal::SoftmaxKernel(context, 128, 32), "tg128_tpr32" },
            // KernelAndBackend{ llmetal::SoftmaxKernel(context, 256, 32), "tg256_tpr32" },
            // KernelAndBackend{ llmetal::SoftmaxKernel(context, 512, 32), "tg512_tpr32" },
            // KernelAndBackend{ llmetal::SoftmaxKernel(context, 1024, 32), "tg1024_tpr32" },
            // KernelAndBackend{ llmetal::SoftmaxKernel(context, 64, 64), "tg64_tpr64" },
            // KernelAndBackend{ llmetal::SoftmaxKernel(context, 128, 128), "tg128_tpr128" },
            // KernelAndBackend{ llmetal::SoftmaxKernel(context, 256, 256), "tg256_tpr256" },
            // KernelAndBackend{ llmetal::SoftmaxKernel(context, 512, 512), "tg512_tpr512" }, // Best
            // KernelAndBackend{ llmetal::SoftmaxKernel(context, 1024, 1024), "tg1024_tpr1024" },
        };
        constexpr std::size_t NUM_KERNELS = sizeof(kernels) / sizeof(KernelAndBackend);

        BenchmarkConfig details[] = {
            BenchmarkConfig{ "SmolLM2-30Kx576", {30000, 576}, {30000} },
            BenchmarkConfig{ "Qwen3.6-3Kx5120", {3000, 5120}, {3000} }
        };

        for (BenchmarkConfig const &config : details) {
            std::cout << "=== " << config.name << " ===" << std::endl;

            // === Preparation ===
            // Get all shapes
            std::uint32_t rows = checked_u32(config.logitsShape[0], "rows");
            std::uint32_t sequence_length = checked_u32(config.logitsShape[1], "sequence_length");
            llmetal::Shape inputShape(config.logitsShape);
            llmetal::Shape validShape(config.validShape);
            llmetal::Shape expectShape(config.logitsShape);

            // Generate data
            std::vector<float> _logits(rows * sequence_length);
            std::vector<std::uint32_t> _valid(rows);
            std::mt19937 rng(rows * sequence_length);
            std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
            std::generate(_logits.begin(), _logits.end(), [&](){ return dist(rng); });
            std::uniform_int_distribution<std::uint32_t> dist2(0, sequence_length - 1);
            std::generate(_valid.begin(), _valid.end(), [&](){ return dist2(rng) ; });

            // Create tensors and fixtures
            llmetal::CpuTensor<float> logits_cpu(inputShape, _logits);
            llmetal::CpuTensor<std::uint32_t> valid_cpu(validShape, _valid);
            llmetal::CpuTensor<float> expect_cpu(expectShape);

            llmetal::GpuTensor<float> logits_gpu = context.upload<float>(logits_cpu);
            llmetal::GpuTensor<std::uint32_t> valid_gpu = context.upload<std::uint32_t>(valid_cpu);
            
            // Validation config
            llmetal::cpu::softmax(logits_cpu, valid_cpu, expect_cpu);
            ValidationFixture fixture {
                config.name,
                inputShape,
                validShape,
                logits_gpu,
                valid_gpu,
                expect_cpu
            };
            // Benchmark config
            BenchmarkConfigWithData config_with_data = {
                config.name,
                inputShape,
                validShape,
                logits_gpu,
                valid_gpu
            };

            // === Validation pass ===
            // Validate the implementations
            std::vector<ValidationResult> validationResults;
            for (std::size_t i = 0; i < NUM_KERNELS; ++i) {
                validationResults.push_back(
                    Validate::run(
                        context, kernels[i].kernel, fixture, kernels[i].backend
                    )
                );
            }
            
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
            std::vector<BenchmarkResult> benchmarkResults;
            for (std::size_t i = 0; i < NUM_KERNELS; ++i) {
                benchmarkResults.push_back(
                    Benchmark::run(
                        context, kernels[i].kernel, config_with_data, kernels[i].backend
                    )
                );
            }

            // Print benchmark results
            benchmarkResults[0].printHeader();
            for (auto const &result : benchmarkResults) {
                result.printRow();
            }
            
            std::cout << std::endl;
        }
        
    } catch (const std::exception& error) {
        std::cerr << "RMS Norm Benchmark failed: " << error.what() << '\n';
        return 1;
    }
}
