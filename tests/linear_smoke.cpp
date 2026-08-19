#include <cstdint>
#include <llmetal/tensor.hpp>
#include "llmetal/linear.hpp"
#include "llmetal/metal_context.hpp"
#include "llmetal/cpu/linear.hpp"
#include "llmetal/verification/equality.hpp"
#include "llmetal/io/reader.hpp"

#include <iostream>
#include <vector>


int main() {
    try {
        const std::filesystem::path fixture_directory = "./tests/fixtures/linear";

        // Shapes from manifest.json:
        // input:    [2, 3, 4]
        // weight:   [6, 4]
        // bias:    [6]
        // expected: [2, 3, 6]
        
        constexpr std::size_t batch_size = 2;
        constexpr std::size_t sequence_length = 3;
        constexpr std::size_t input_hidden_size = 4;
        constexpr std::size_t output_hidden_size = 6;
        const std::uint32_t token_count = checked_u32(checked_multiply(batch_size, sequence_length), "token_count");
        const std::uint32_t weight_count = checked_u32(checked_multiply(input_hidden_size, output_hidden_size), "expected_count");
        const std::uint32_t input_count = checked_u32(checked_multiply(token_count, input_hidden_size), "expected_count");
        const std::uint32_t expected_count = checked_u32(checked_multiply(token_count, output_hidden_size), "expected_count");

        // Get fixtures
        llmetal::CpuTensor<float> input_tensor_cpu(
            llmetal::Shape{ batch_size, sequence_length, input_hidden_size }, 
            read_raw<float>(fixture_directory / "x.f32", input_count)
        );
        llmetal::CpuTensor<float> weight_tensor_cpu(
            llmetal::Shape{ output_hidden_size, input_hidden_size }, 
            read_raw<float>(fixture_directory / "weight.f32", weight_count)
        );
        llmetal::CpuTensor<float> bias_tensor_cpu(
            llmetal::Shape{ output_hidden_size }, 
            read_raw<float>(fixture_directory / "bias.f32", output_hidden_size)
        );

        // Oracle with bias
        llmetal::CpuTensor<float> expected_tensor_with_bias_cpu(
            llmetal::Shape{ batch_size, sequence_length, output_hidden_size }, 
            read_raw<float>(fixture_directory / "expected.f32", expected_count)
        );
        
        // Oracle without bias
        // std::vector<float> _expected_without_bias(expected_count);
        llmetal::CpuTensor<float> expected_tensor_without_bias_cpu(
            llmetal::Shape{ batch_size, sequence_length, output_hidden_size });
        for (std::size_t b = 0; b < batch_size; ++b) {
            for (std::size_t s = 0; s < sequence_length; ++s) {
                for (std::size_t o = 0; o < output_hidden_size; ++o) {
                    expected_tensor_without_bias_cpu[b * sequence_length * output_hidden_size + s * output_hidden_size + o] 
                        = expected_tensor_with_bias_cpu[b * sequence_length * output_hidden_size + s * output_hidden_size + o] - bias_tensor_cpu[o];
                }
            }
        }

        // Verify cpu oracle
        llmetal::CpuTensor<float> output_tensor_with_bias_cpu(
            llmetal::Shape{batch_size, sequence_length, output_hidden_size}
        );
        llmetal::cpu::linear(input_tensor_cpu, weight_tensor_cpu, bias_tensor_cpu, output_tensor_with_bias_cpu);

        llmetal::CpuTensor<float> output_tensor_without_bias_cpu(
            llmetal::Shape{batch_size, sequence_length, output_hidden_size}
        );
        llmetal::cpu::linear(input_tensor_cpu, weight_tensor_cpu, output_tensor_without_bias_cpu);

        // Verify cpu oracle.
        bool result = verify_equal(output_tensor_with_bias_cpu, expected_tensor_with_bias_cpu) &&
                      verify_equal(output_tensor_without_bias_cpu, expected_tensor_without_bias_cpu);

        if (!result) {
            std::cerr << "Linear test failed - cpu oracle not matching." << '\n';
            return 1;
        }

        llmetal::MetalContext context;
        llmetal::LinearKernel kernel(context, 32, 1);
        auto input_tensor_gpu = context.upload(input_tensor_cpu);
        auto weight_tensor_gpu = context.upload(weight_tensor_cpu);
        auto bias_tensor_gpu = context.upload(bias_tensor_cpu);
        
        // Verify with bias
        auto output_tensor_with_bias_gpu = context.allocate<float>(
            llmetal::Shape{ batch_size, sequence_length, output_hidden_size }
        );
        auto job = kernel.submit(
            input_tensor_gpu, weight_tensor_gpu, &bias_tensor_gpu, output_tensor_with_bias_gpu
        );
        job.wait();
        auto output_tensor_with_bias_gpu_cpu = context.download(output_tensor_with_bias_gpu);
        result = verify_equal(output_tensor_with_bias_gpu_cpu, expected_tensor_with_bias_cpu);

        if (!result) {
            std::cerr << "Linear test failed for bias." << '\n';
            return 1;
        }

        // Verify without bias
        auto output_tensor_without_bias_gpu = context.allocate<float>(
            llmetal::Shape{ batch_size, sequence_length, output_hidden_size }
        );
        job = kernel.submit(
            input_tensor_gpu, weight_tensor_gpu, nullptr, output_tensor_without_bias_gpu
        );
        job.wait();
        auto output_tensor_without_bias_gpu_cpu = context.download(output_tensor_without_bias_gpu);
        result = verify_equal(output_tensor_without_bias_gpu_cpu, expected_tensor_without_bias_cpu);

        if (!result) {
            std::cerr << "Linear test failed for without bias." << '\n';
            return 1;
        }

    } catch (const std::exception& error) {
        std::cerr << "Linear test failed: " << error.what() << '\n';
        return 1;
    }

    return 0;
}
