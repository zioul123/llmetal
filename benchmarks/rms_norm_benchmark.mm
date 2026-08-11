#include "llmetal/cpu/rms_norm.hpp"
#include "llmetal/rms_norm.hpp"
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
    llmetal::Shape inputShape;
    llmetal::Shape weightsShape;
};

struct BenchmarkConfigWithData {
    std::string_view name;
    llmetal::Shape inputShape;
    llmetal::Shape weightsShape;
    const llmetal::GpuTensor<float>& input;
    const llmetal::GpuTensor<float>& weights;
    const float epsilon;

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
            << std::setw(shape_width) << config.inputShape.to_string()
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
    llmetal::Shape inputShape;
    llmetal::Shape weightsShape;
    const llmetal::GpuTensor<float>& inputs;
    const llmetal::GpuTensor<float>& weights;
    const float epsilon;
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
    llmetal::RmsNormKernel& kernel, 
    const BenchmarkConfigWithData& config, 
    std::string_view backend
) {
    if (config.timed_runs == 0) throw std::invalid_argument("timed_runs must be greater than 0");
    std::uint32_t batch_size = checked_u32(config.inputShape[0], "batch_size");
    std::uint32_t sequence_length = checked_u32(config.inputShape[1], "sequence_length");
    std::uint32_t hidden_size = checked_u32(config.weightsShape[0], "hidden_size");

    // Prepare output tensor
    auto output_tensor_gpu = context.allocate<float>(
        llmetal::Shape{batch_size, sequence_length, hidden_size}
    );
    
    // Run warmup rounds
    if (config.warmup_runs != 0) {
        auto job = kernel.submit_repeated(
            config.warmup_runs, 
            config.input, 
            config.weights, 
            output_tensor_gpu,
            config.epsilon
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
                config.input, 
                config.weights, 
                output_tensor_gpu,
                config.epsilon
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
            config.input, 
            config.weights, 
            output_tensor_gpu,
            config.epsilon
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
    llmetal::Shape inputShape,
    llmetal::Shape weightsShape,
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
            inputShape,
            weightsShape,
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
                inputShape,
                weightsShape,
                false,
                message.str(),
            };
        }
    }

    return {
        std::string(backend),
        inputShape,
        weightsShape,
        true,
        {},
    };
}

ValidationResult run(
    llmetal::MetalContext& context, 
    llmetal::RmsNormKernel& kernel,
    const ValidationFixture& fixture,
    std::string_view backend
) {
    std::uint32_t batch_size = checked_u32(fixture.inputShape[0], "batch_size");
    std::uint32_t sequence_length = checked_u32(fixture.inputShape[1], "sequence_length");
    std::uint32_t hidden_size = checked_u32(fixture.weightsShape[0], "hidden_size");
    // Prepare output tensor
    auto output_tensor_gpu = context.allocate<float>(
        llmetal::Shape{batch_size, sequence_length, hidden_size}
    );
    // Run the kernel
    auto job = kernel.submit(
        fixture.inputs, 
        fixture.weights, 
        output_tensor_gpu, 
        fixture.epsilon
    );
    job.wait();
    auto output_tensor_cpu = context.download(output_tensor_gpu);

    // Validate
    return validate_output(
        std::string(backend), 
        fixture.inputShape,
        fixture.weightsShape,
        output_tensor_cpu,
        fixture.expect
    );
}

} // namespace Validate

struct KernelAndBackend {
    llmetal::RmsNormKernel kernel;
    std::string_view backend;
};

int main() {
    try {
        llmetal::MetalContext context;
        KernelAndBackend kernels[] = {
            KernelAndBackend{ llmetal::RmsNormKernel(context, 32, 1), "tg32_tpr1" },
            KernelAndBackend{ llmetal::RmsNormKernel(context, 32, 32), "tg32_tpr32" },
            KernelAndBackend{ llmetal::RmsNormKernel(context, 64, 32), "tg64_tpr32" },
            KernelAndBackend{ llmetal::RmsNormKernel(context, 128, 32), "tg128_tpr32" },
            KernelAndBackend{ llmetal::RmsNormKernel(context, 256, 32), "tg256_tpr32" },
            KernelAndBackend{ llmetal::RmsNormKernel(context, 512, 32), "tg512_tpr32" },
            KernelAndBackend{ llmetal::RmsNormKernel(context, 1024, 32), "tg1024_tpr32" },
        };
        constexpr std::size_t NUM_KERNELS = sizeof(kernels) / sizeof(KernelAndBackend);

        BenchmarkConfig details[] = {
            BenchmarkConfig{ "SmolLM2-1000x576", {30, 1000, 576}, {576} }
        };

        for (BenchmarkConfig const &config : details) {
            std::cout << "=== " << config.name << " ===" << std::endl;

            // === Preparation ===
            // Get all shapes
            std::uint32_t batch_size = checked_u32(config.inputShape[0], "batch_size");
            std::uint32_t sequence_length = checked_u32(config.inputShape[1], "sequence_length");
            std::uint32_t hidden_size = checked_u32(config.weightsShape[0], "hidden_size");
            llmetal::Shape inputShape(config.inputShape);
            llmetal::Shape weightsShape(config.weightsShape);
            llmetal::Shape expectShape(config.inputShape);

            // Generate data
            std::vector<float> _inputs(batch_size * sequence_length * hidden_size);
            std::vector<float> _weights(hidden_size);
            std::mt19937 rng(batch_size * sequence_length * hidden_size);
            std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
            std::generate(_inputs.begin(), _inputs.end(), [&](){ return dist(rng); });
            std::generate(_weights.begin(), _weights.end(), [&](){ return dist(rng) ; });
            float epsilon = 0.0001f;

            // Create tensors and fixtures
            llmetal::CpuTensor<float> inputs_cpu(inputShape, _inputs);
            llmetal::CpuTensor<float> weights_cpu(weightsShape, _weights);
            llmetal::CpuTensor<float> expect_cpu(expectShape);

            llmetal::GpuTensor<float> inputs_gpu = context.upload<float>(inputs_cpu);
            llmetal::GpuTensor<float> weights_gpu = context.upload<float>(weights_cpu);
            
            // Validation config
            llmetal::cpu::rms_norm(inputs_cpu, weights_cpu, expect_cpu, epsilon);
            ValidationFixture fixture {
                config.name,
                inputShape,
                weightsShape,
                inputs_gpu,
                weights_gpu,
                epsilon,
                expect_cpu
            };
            // Benchmark config
            BenchmarkConfigWithData config_with_data = {
                config.name,
                inputShape,
                weightsShape,
                inputs_gpu,
                weights_gpu,
                epsilon
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
