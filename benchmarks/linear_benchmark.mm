#include "llmetal/cpu/linear.hpp"
#include "llmetal/linear.hpp"
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
    llmetal::Shape inputShape;  // [batch_size, seq_length, input_hidden_size]
    llmetal::Shape weightShape; // [output_hidden_size, input_hidden_size]
    llmetal::Shape biasShape;   // [output_hidden_size]
};

struct BenchmarkConfigWithData {
    std::string_view name;
    llmetal::Shape inputShape;  // [batch_size, seq_length, input_hidden_size]
    llmetal::Shape weightShape; // [output_hidden_size, input_hidden_size]
    llmetal::Shape biasShape;   // [output_hidden_size]
    const llmetal::GpuTensor<float>& input;
    const llmetal::GpuTensor<float>& weight;
    const llmetal::GpuTensor<float>& bias;

    std::size_t warmup_runs = 5;
    std::size_t timed_runs = 10;
    std::size_t repeats = 10; // NOT USED YET
};

constexpr int backend_width = 20;
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
    llmetal::Shape inputShape;  // [batch_size, seq_length, input_hidden_size]
    llmetal::Shape weightShape; // [output_hidden_size, input_hidden_size]
    llmetal::Shape biasShape;   // [output_hidden_size]
    bool withBias;
    const llmetal::GpuTensor<float>& input;
    const llmetal::GpuTensor<float>& weight;
    const llmetal::GpuTensor<float>& bias;
    const llmetal::CpuTensor<float>& expect;
};

struct ValidationResult {
    std::string backend;
    llmetal::Shape inputShape;  // [batch_size, seq_length, input_hidden_size]
    llmetal::Shape weightShape; // [output_hidden_size, input_hidden_size]
    llmetal::Shape biasShape;   // [output_hidden_size]
    bool valid;
    std::string error;
};

namespace Benchmark {

BenchmarkResult run(
    llmetal::MetalContext& context, 
    llmetal::LinearKernel& kernel, 
    const BenchmarkConfigWithData& config, 
    std::string_view backend,
    bool with_bias
) {
    if (config.timed_runs == 0) throw std::invalid_argument("timed_runs must be greater than 0");
    std::uint32_t batch_size = checked_u32(config.inputShape[0], "batch_size");
    std::uint32_t sequence_length = checked_u32(config.inputShape[1], "sequence_length");
    std::uint32_t output_hidden_size = checked_u32(config.weightShape[0], "output_hidden_size");
    llmetal::Shape outputShape{batch_size, sequence_length, output_hidden_size};
    // Prepare output tensor
    auto output_tensor_gpu = context.allocate<float>(outputShape);
    
    // Run warmup rounds
    if (config.warmup_runs != 0) {
        if (with_bias) {
            auto job = kernel.submit_repeated(
                config.warmup_runs, config.input, 
                config.weight, &config.bias,
                output_tensor_gpu
            );
            job.wait();
        } else {
            auto job = kernel.submit_repeated(
                config.warmup_runs, config.input, 
                config.weight, nullptr, output_tensor_gpu
            );
            job.wait();
        }
        auto result = context.download(output_tensor_gpu);
    }

    // Run timed rounds for single host time
    std::vector<std::chrono::nanoseconds> e2eDurations;
    {
        for (std::size_t i = 0; i < config.timed_runs; ++i) {
            auto start = now();
            if (with_bias) {
                auto job = kernel.submit_repeated(
                    config.timed_runs, config.input, 
                    config.weight, &config.bias,
                    output_tensor_gpu
                );
                job.wait();
            } else {
                auto job = kernel.submit_repeated(
                    config.timed_runs, config.input, 
                    config.weight, nullptr, output_tensor_gpu
                );
                job.wait();
            }
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
        if (with_bias) {
            auto job = kernel.submit_repeated(
                config.timed_runs, config.input, 
                config.weight, &config.bias,
                output_tensor_gpu
            );
            job.wait();
            batchGpuDuration = job.gpu_duration() / config.timed_runs;
        } else {
            auto job = kernel.submit_repeated(
                config.timed_runs, config.input, 
                config.weight, nullptr, output_tensor_gpu
            );
            job.wait();
            batchGpuDuration = job.gpu_duration() / config.timed_runs;
        }
        auto stop = now();
    
        batchHostDuration = (stop - start) / config.timed_runs;
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
    llmetal::Shape inputShape,  // [batch_size, seq_length, input_hidden_size]
    llmetal::Shape weightShape, // [output_hidden_size, input_hidden_size]
    llmetal::Shape biasShape,   // [output_hidden_size]
    const llmetal::CpuTensor<float>& actual,
    const llmetal::CpuTensor<float>& expect
) {
    constexpr float absoluteTolerance = 1.0e-3f;
    constexpr float relativeTolerance = 1.0e-4f;
    
    if (expect.numel() != actual.numel()) {
        std::ostringstream message;
        message
            << "actual numel = " << actual.numel()
            << ", expect numel = " << expect.numel();
        return {
            std::string(backend),
            inputShape,
            weightShape,
            biasShape,
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

            // std::cout << "Actual"<< std::endl;
            // std::cout << actual << std::endl;
            // std::cout << "Expect"<< std::endl;
            // std::cout << expect << std::endl;
            return {
                std::string(backend),
                inputShape,
                weightShape,
                biasShape,
                false,
                message.str(),
            };
        }
    }

    return {
        std::string(backend),
        inputShape,
        weightShape,
        biasShape,
        true,
        {},
    };
}

ValidationResult run(
    llmetal::MetalContext& context, 
    llmetal::LinearKernel& kernel,
    const ValidationFixture& fixture,
    std::string_view backend
) {
    // Prepare output tensor
    std::uint32_t batch_size = checked_u32(fixture.inputShape[0], "batch_size");
    std::uint32_t sequence_length = checked_u32(fixture.inputShape[1], "sequence_length");
    std::uint32_t output_hidden_size = checked_u32(fixture.weightShape[0], "output_hidden_size");
    llmetal::Shape outputShape{batch_size, sequence_length, output_hidden_size};
    auto output_tensor_gpu = context.allocate<float>(outputShape);
    // Run the kernel
    if (fixture.withBias) {
        auto job = kernel.submit(
            fixture.input, fixture.weight,
            &fixture.bias, output_tensor_gpu
        );
        job.wait();
    } else {
        auto job = kernel.submit(
            fixture.input, fixture.weight,
            nullptr, output_tensor_gpu
        );
        job.wait();
    }
    auto output_tensor_cpu = context.download(output_tensor_gpu);

    // Validate
    return validate_output(
        std::string(backend), 
        fixture.inputShape,
        fixture.weightShape,
        fixture.biasShape,
        output_tensor_cpu,
        fixture.expect
    );
}

} // namespace Validate

struct KernelAndBackend {
    llmetal::LinearKernel kernel;
    std::string_view backend;
};

int main() {
    try {
        llmetal::MetalContext context;
        KernelAndBackend kernels[] = {
            KernelAndBackend{ llmetal::LinearKernel(context, 32, 1), "tg32_tpr1_rpt1" },
            KernelAndBackend{ llmetal::LinearKernel(context, 32, 1, 2), "tg32_tpr1_rpt2" },
            KernelAndBackend{ llmetal::LinearKernel(context, 32, 1, 4), "tg32_tpr1_rpt4" },
            KernelAndBackend{ llmetal::LinearKernel(context, 32, 1, 8), "tg32_tpr1_rpt8" },
            KernelAndBackend{ llmetal::LinearKernel(context, 32, 32), "tg32_tpr32" },
            KernelAndBackend{ llmetal::LinearKernel(context, 64, 32), "tg64_tpr32" },
            KernelAndBackend{ llmetal::LinearKernel(context, 128, 32), "tg128_tpr32" },
            KernelAndBackend{ llmetal::LinearKernel(context, 256, 32), "tg256_tpr32" },
            KernelAndBackend{ llmetal::LinearKernel(context, 512, 32), "tg512_tpr32" },
            KernelAndBackend{ llmetal::LinearKernel(context, 1024, 32), "tg1024_tpr32" },
            KernelAndBackend{ llmetal::LinearKernel(context, 64, 64), "tg64_tpr64" },
            KernelAndBackend{ llmetal::LinearKernel(context, 128, 128), "tg128_tpr128" },
            KernelAndBackend{ llmetal::LinearKernel(context, 256, 256), "tg256_tpr256" },
            KernelAndBackend{ llmetal::LinearKernel(context, 512, 512), "tg512_tpr512" },
            KernelAndBackend{ llmetal::LinearKernel(context, 1024, 1024), "tg1024_tpr1024" },
        };
        constexpr std::size_t NUM_KERNELS = sizeof(kernels) / sizeof(KernelAndBackend);

        BenchmarkConfig details[] = {
            // BenchmarkConfig{ "SmolLM2-30Kx576-QProj", {1, 1, 64}, {32, 64}, llmetal::Shape{32}},
            BenchmarkConfig{ "SmolLM2-30Kx576-QProj", {5, 30, 576}, {576, 576}, llmetal::Shape{576}},
            // BenchmarkConfig{ "SmolLM2-30Kx576-QProj", {5, 30, 576}, {576, 576}, llmetal::Shape{576}},
            // BenchmarkConfig{ "SmolLM2-30Kx576-KVProj", {5, 30, 576}, {192, 576}, llmetal::Shape{192}},
            BenchmarkConfig{ "Qwen3.6-3Kx5120-attention-QProj", {2, 4, 5120 }, {6144, 5120}, llmetal::Shape{6144} },
            // BenchmarkConfig{ "Qwen3.6-3Kx5120-attention-KVProj", {3, 10, 5120}, {1024, 5120}, llmetal::Shape{1024} },
            // BenchmarkConfig{ "Qwen3.6-3Kx5120-deltanet-QKProj", {3, 10, 5120}, {2048, 5120}, llmetal::Shape{2048} }
        };

        for (BenchmarkConfig const &config : details) {
            std::cout << "=== " << config.name << " ===" << std::endl;

            // === Preparation ===
            // Get all shapes
            std::uint32_t batch_size = checked_u32(config.inputShape[0], "batch_size");
            std::uint32_t sequence_length = checked_u32(config.inputShape[1], "sequence_length");
            std::uint32_t output_hidden_size = checked_u32(config.weightShape[0], "output_hidden_size");
            std::uint32_t input_hidden_size = checked_u32(config.weightShape[1], "input_hidden_size");
            llmetal::Shape outputShape{batch_size, sequence_length, output_hidden_size};
            // Generate data
            std::mt19937 rng(batch_size * sequence_length * output_hidden_size * input_hidden_size);
            std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
            
            std::vector<float> _input(batch_size * sequence_length * input_hidden_size);
            std::generate(_input.begin(), _input.end(), [&](){ return dist(rng); });

            std::vector<float> _weight(output_hidden_size * input_hidden_size);
            std::generate(_weight.begin(), _weight.end(), [&](){ return dist(rng); });

            std::vector<float> _bias(output_hidden_size);
            std::generate(_bias.begin(), _bias.end(), [&](){ return dist(rng); });

            // Create tensors and fixtures
            llmetal::CpuTensor<float> input_cpu(config.inputShape, _input);
            llmetal::CpuTensor<float> weight_cpu(config.weightShape, _weight);
            llmetal::CpuTensor<float> bias_cpu(config.biasShape, _bias);
            llmetal::CpuTensor<float> expect_cpu_with_bias(outputShape);
            llmetal::CpuTensor<float> expect_cpu_without_bias(outputShape);
            llmetal::cpu::linear(input_cpu, weight_cpu, bias_cpu, expect_cpu_with_bias);
            llmetal::cpu::linear(input_cpu, weight_cpu, expect_cpu_without_bias);

            llmetal::GpuTensor<float> input_gpu = context.upload<float>(input_cpu);
            llmetal::GpuTensor<float> weight_gpu = context.upload<float>(weight_cpu);
            llmetal::GpuTensor<float> bias_gpu = context.upload<float>(bias_cpu);
            
            // Validation config
            ValidationFixture fixture_with_bias {
                config.name, config.inputShape, config.weightShape, config.biasShape, 
                true, input_gpu, weight_gpu, bias_gpu, expect_cpu_with_bias
            };
            ValidationFixture fixture_without_bias {
                config.name, config.inputShape, config.weightShape, config.biasShape, 
                false, input_gpu, weight_gpu, bias_gpu, expect_cpu_without_bias
            };
            // Benchmark config
            BenchmarkConfigWithData config_with_data = {
                config.name, config.inputShape, config.weightShape, config.biasShape, 
                input_gpu, weight_gpu, bias_gpu
            };

            for (auto validationFixture: {fixture_with_bias, fixture_without_bias}) {
                std::cout << "=== With Bias:" << validationFixture.withBias << std::endl;

                // === Validation pass ===
                // Validate the implementations
                std::vector<ValidationResult> validationResults;
                for (std::size_t i = 0; i < NUM_KERNELS; ++i) {
                    validationResults.push_back(
                        Validate::run(
                            context, kernels[i].kernel, validationFixture, kernels[i].backend
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
                            context, kernels[i].kernel, 
                            config_with_data, kernels[i].backend, 
                            validationFixture.withBias
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
        }
        
    } catch (const std::exception& error) {
        std::cerr << "Linear Benchmark failed: " << error.what() << '\n';
        return 1;
    }
}
