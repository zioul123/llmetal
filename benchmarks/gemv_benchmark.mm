#include "llmetal/gemv_mps.hpp"
#include "llmetal/tensor.hpp"
#include "llmetal/metal_context.hpp"
#include "llmetal/gemv_naive.hpp"
#include "llmetal/gemv_NRPSG.hpp"
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

#include <Foundation/Foundation.h>
#include <Metal/Metal.h>
#include <MetalPerformanceShaders/MetalPerformanceShaders.h>

// Aliases for brevity
using us = std::chrono::microseconds;
using ns = std::chrono::nanoseconds;
auto now = std::chrono::steady_clock::now;

template <typename K>
concept GemvKernel = requires(
    K& kernel,
    llmetal::Shape shape,
    const llmetal::GpuTensor<float>& matrix,
    const llmetal::GpuTensor<float>& vector,
    llmetal::GpuTensor<float>& output,
    std::size_t repeats
) {
    { kernel.submit(matrix, vector, output) } -> std::same_as<llmetal::MetalJob>;
    { kernel.submit_repeated(repeats, matrix, vector, output) } -> std::same_as<llmetal::MetalJob>;
};

struct BenchmarkConfig {
    std::string_view name;
    llmetal::Shape shape;
};

struct BenchmarkConfigWithData {
    std::string_view name;
    llmetal::Shape shape;
    llmetal::GpuTensor<float> matrix;
    llmetal::GpuTensor<float> vector;

    std::size_t warmup_runs = 20;
    std::size_t timed_runs = 100;
};

struct MpsBenchmarkConfigWithData {
    std::string_view name;
    llmetal::Shape shape;
    std::vector<float> matrix;
    std::vector<float> vector;

    std::size_t warmup_runs = 20;
    std::size_t timed_runs = 100;
};

constexpr int backend_width = 14;
constexpr int shape_width   = 12;
constexpr int time_width    = 16;
constexpr int total_time_width = 19;  // 12 digits + " us"
struct BenchmarkResult {
    std::string backend;
    llmetal::Shape shape;

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
            << (std::to_string(shape[0]) + "x" +
                std::to_string(shape[1]))
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
    llmetal::Shape shape;
    const llmetal::GpuTensor<float>& matrix;
    const llmetal::GpuTensor<float>& vector;
    const std::span<float>& expect;
};

struct MpsValidationFixture {
    std::string_view name;
    llmetal::Shape shape;
    const std::vector<float>& matrix;
    const std::vector<float>& vector;
    const std::span<float>& expect;
};

struct ValidationResult {
    std::string backend;
    llmetal::Shape shape;

    bool valid;
    std::string error;
};

namespace Benchmark {

template <GemvKernel Kernel>
BenchmarkResult run(llmetal::MetalContext& context, Kernel& kernel, BenchmarkConfigWithData config, std::string_view backend) {
    if (config.timed_runs == 0) throw std::invalid_argument("timed_runs must be greater than 0");

    // Prepare kernel
    llmetal::GpuTensor<float> output = context.allocate<float>(config.shape);
        
    // Run warmup rounds
    if (config.warmup_runs != 0) {
        auto job = kernel.submit_repeated(
            config.warmup_runs, config.matrix, config.vector, output
        );
        job.wait();
        llmetal::CpuTensor<float> output_cpu = context.download(output);
    }

    // Run timed rounds for single host time
    std::vector<std::chrono::nanoseconds> e2eDurations;
    {
        for (std::size_t i = 0; i < config.timed_runs; ++i) {
            auto start = now();
            auto job = kernel.submit(config.matrix, config.vector, output);
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
        auto job = kernel.submit_repeated(
            config.timed_runs, config.matrix, config.vector, output
        );
        job.wait();
        auto stop = now();
    
        batchHostDuration = (stop - start) / config.timed_runs;
        batchGpuDuration = job.gpu_duration() / config.timed_runs;
    }

    return {
        .backend = std::string(backend),
        .shape = config.shape,
        .single_host_min = host_min,
        .single_host_median = host_median,
        .single_host_p95 = host_p95,
        .batched_host_per_dispatch = batchHostDuration,
        .batched_gpu_per_dispatch = batchGpuDuration,
    };
}

BenchmarkResult runMps(llmetal::MetalContext& context, llmetal::GemvMpsKernel& kernel, MpsBenchmarkConfigWithData config, std::string_view backend) {
    if (config.timed_runs == 0) throw std::invalid_argument("timed_runs must be greater than 0");

    // Prepare kernel
    std::vector<float> output(config.shape[0]);
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
        .shape = config.shape,
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
    const llmetal::Shape& shape,
    const std::span<float>& actual,
    const std::span<float>& expect
) {
    // constexpr float absoluteTolerance = 1.0e-4f;
    constexpr float absoluteTolerance = 1.0e-3f;
    constexpr float relativeTolerance = 1.0e-4f;
    // constexpr float relativeTolerance = 1.0e-5f;
    if (actual.size() != expect.size()) { 
        std::ostringstream message;
        message
            << "Actual size: " << actual.size()
            << ", expected size: " << expect.size();
        return {
            .backend = std::string(backend),
            .shape = shape,
            .valid = false,
            .error = message.str(),
        };
    }
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
ValidationResult run(llmetal::MetalContext& context, Kernel& kernel, const ValidationFixture& fixture, std::string_view backend) {
    // Run the kernel
    llmetal::GpuTensor<float> output = context.allocate<float>({fixture.shape[0]});
    auto job = kernel.submit(fixture.matrix, fixture.vector, output);
    job.wait();
    llmetal::CpuTensor<float> output_cpu = context.download(output);

    // Validate
    return validate_output(
        std::string(backend), 
        fixture.shape, 
        output_cpu.span(), 
        fixture.expect
    );
}

ValidationResult runMps(llmetal::MetalContext& context, llmetal::GemvMpsKernel& kernel, const MpsValidationFixture& fixture, std::string_view backend) {
    std::uint32_t rows = checked_u32(fixture.shape[0], "matrix_rows");

    std::vector<float> output(fixture.shape[0]);
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
        llmetal::GemvNRPSGKernel oneRPSGKernel(context, 1, 1);
        llmetal::GemvNRPSGKernel twoRPSGKernel(context, 2, 1);
        llmetal::GemvNRPSGKernel fourRPSGKernel(context, 4, 1);
        llmetal::GemvNRPSGKernel eightRPSGKernel(context, 8, 1);
        llmetal::GemvNRPSGKernel oneRPSG_2SGPTG_Kernel(context, 1, 2);
        llmetal::GemvNRPSGKernel twoRPSG_2SGPTG_Kernel(context, 2, 2);
        llmetal::GemvNRPSGKernel fourRPSG_2SGPTG_Kernel(context, 4, 2);
        llmetal::GemvNRPSGKernel eightRPSG_2SGPTG_Kernel(context, 8, 2);
        llmetal::GemvNRPSGKernel oneRPSG_4SGPTG_Kernel(context, 1, 4);
        llmetal::GemvNRPSGKernel twoRPSG_4SGPTG_Kernel(context, 2, 4);
        llmetal::GemvNRPSGKernel fourRPSG_4SGPTG_Kernel(context, 4, 4);
        llmetal::GemvNRPSGKernel eightRPSG_4SGPTG_Kernel(context, 8, 4);
        llmetal::GemvNRPSGKernel oneRPSG_8SGPTG_Kernel(context, 1, 8);
        llmetal::GemvNRPSGKernel twoRPSG_8SGPTG_Kernel(context, 2, 8);
        llmetal::GemvNRPSGKernel fourRPSG_8SGPTG_Kernel(context, 4, 8);
        llmetal::GemvNRPSGKernel eightRPSG_8SGPTG_Kernel(context, 8, 8);
        
        BenchmarkConfig details[] = {
            // { "1024x1024", 1024, 1024},
            // { "2048x1024", 2048, 1024},
            // { "1024x2048", 1024, 2048},
            // { "3072x1024", 3072, 1024},
            // { "1024x3072", 1024, 3072},
            { "4096x4096", {4096, 4096}},
            { "8192x8192", {8192, 8192}},
            { "11008x4096", {11008, 4096}},
            { "4096x11008", {4096, 11008}},
        };

        for (BenchmarkConfig const &config : details) {
            std::cout << "=== " << config.name << " ===" << std::endl;

            // === Preparation ===

            // Generate random data
            std::size_t element_count = config.shape[1] * config.shape[0];
            std::vector<float> matrix(element_count);
            std::vector<float> vector(config.shape[1]);
            std::vector<float> expect(config.shape[0]);
          
            std::mt19937 rng(element_count);
            std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
            std::generate(matrix.begin(), matrix.end(), [&](){ return dist(rng); });
            std::generate(vector.begin(), vector.end(), [&](){ return dist(rng); });

            llmetal::GpuTensor<float> matrix_gpu = context.upload<float>(config.shape, matrix);
            llmetal::GpuTensor<float> vector_gpu = context.upload<float>({ config.shape[1] }, vector);

            BenchmarkConfigWithData config_with_data = {
                config.name, config.shape, matrix_gpu, vector_gpu
            };

            // === Validation pass ===

            // Produce CPU result and fixture
            llmetal::cpu::gemv_f32(config_with_data.shape, matrix, vector, expect);
            ValidationFixture fixture { 
                config_with_data.name,
                config_with_data.shape,
                matrix_gpu,
                vector_gpu,
                expect
            };

            // Validate the implementations
            ValidationResult validationResults[] = {
                Validate::run(context, naiveKernel,     fixture, "naive"),

                Validate::run(context, oneRPSGKernel,  fixture, "1rpsg"),
                Validate::run(context, twoRPSGKernel,   fixture, "2rpsg"),
                Validate::run(context, fourRPSGKernel,  fixture, "4rpsg"),
                Validate::run(context, eightRPSGKernel, fixture, "8rpsg"),

                Validate::run(context, oneRPSG_2SGPTG_Kernel,  fixture, "1rpsg_2sgptg"),
                Validate::run(context, twoRPSG_2SGPTG_Kernel,   fixture, "2rpsg_2sgptg"),
                Validate::run(context, fourRPSG_2SGPTG_Kernel,  fixture, "4rpsg_2sgptg"),
                Validate::run(context, eightRPSG_2SGPTG_Kernel, fixture, "8rpsg_2sgptg"),

                Validate::runMps(context, mpsKernel, {
                    config_with_data.name,
                    config_with_data.shape,
                    matrix,
                    vector,
                    expect
                }, "mps"),

                Validate::run(context, oneRPSG_4SGPTG_Kernel,  fixture, "1rpsg_4sgptg"),
                Validate::run(context, twoRPSG_4SGPTG_Kernel,   fixture, "2rpsg_4sgptg"),
                Validate::run(context, fourRPSG_4SGPTG_Kernel,  fixture, "4rpsg_4sgptg"),
                Validate::run(context, eightRPSG_4SGPTG_Kernel, fixture, "8rpsg_4sgptg"),

                Validate::run(context, oneRPSG_8SGPTG_Kernel,  fixture, "1rpsg_8sgptg"),
                Validate::run(context, twoRPSG_8SGPTG_Kernel,   fixture, "2rpsg_8sgptg"),
                Validate::run(context, fourRPSG_8SGPTG_Kernel,  fixture, "4rpsg_8sgptg"),
                Validate::run(context, eightRPSG_8SGPTG_Kernel, fixture, "8rpsg_8sgptg"),
            };
            
            // Print validation results
            std::cout << "Validation:\n";
            bool any_failed = false;
            for (auto const &result : validationResults) {
                if (!result.valid) {
                    std::cerr << "  " << result.backend << ": failed - " << result.error << '\n';
                    any_failed = true;
                }
                // else {
                //     std::cout << "  " << result.backend << ": passed\n";
                // }
            }
            if (any_failed) {
                std::cerr << "Skipping benchmarks because validation failed." << std::endl;
                return 1;
            } else {
                std::cout << "  Validation passed." << std::endl;
            }

            // === Benchmark pass ===
            BenchmarkResult benchmarkResults[] = {
                Benchmark::run(context, naiveKernel,     config_with_data, "naive"),

                Benchmark::run(context, oneRPSGKernel,  config_with_data, "1rpsg"),
                Benchmark::run(context, twoRPSGKernel,   config_with_data, "2rpsg"),
                Benchmark::run(context, fourRPSGKernel,  config_with_data, "4rpsg"),
                Benchmark::run(context, eightRPSGKernel, config_with_data, "8rpsg"),

                Benchmark::run(context, oneRPSG_2SGPTG_Kernel, config_with_data, "1rpsg_2sgptg"),
                Benchmark::run(context, twoRPSG_2SGPTG_Kernel, config_with_data, "2rpsg_2sgptg"),
                Benchmark::run(context, fourRPSG_2SGPTG_Kernel, config_with_data, "4rpsg_2sgptg"),
                Benchmark::run(context, eightRPSG_2SGPTG_Kernel, config_with_data, "8rpsg_2sgptg"),

                Benchmark::runMps(context, mpsKernel, { config_with_data.name, config_with_data.shape, matrix, vector }, "mps"),

                Benchmark::run(context, oneRPSG_4SGPTG_Kernel, config_with_data, "1rpsg_4sgptg"),
                Benchmark::run(context, twoRPSG_4SGPTG_Kernel, config_with_data, "2rpsg_4sgptg"),
                Benchmark::run(context, fourRPSG_4SGPTG_Kernel, config_with_data, "4rpsg_4sgptg"),
                Benchmark::run(context, eightRPSG_4SGPTG_Kernel, config_with_data, "8rpsg_4sgptg"),

                Benchmark::run(context, oneRPSG_8SGPTG_Kernel, config_with_data, "1rpsg_8sgptg"),
                Benchmark::run(context, twoRPSG_8SGPTG_Kernel, config_with_data, "2rpsg_8sgptg"),
                Benchmark::run(context, fourRPSG_8SGPTG_Kernel, config_with_data, "4rpsg_8sgptg"),
                Benchmark::run(context, eightRPSG_8SGPTG_Kernel, config_with_data, "8rpsg_8sgptg"),
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
