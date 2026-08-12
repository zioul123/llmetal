#include <cstdint>
#include <llmetal/tensor.hpp>
#include "llmetal/metal_context.hpp"
#include "llmetal/cpu/rms_norm.hpp"
#include "llmetal/rms_norm.hpp"
#include "llmetal/verification/equality.hpp"
#include "llmetal/io/reader.hpp"

#include <iostream>
#include <vector>


int main() {
    try {
        const std::filesystem::path fixture_directory = "./tests/fixtures/rms_norm";

        // Shapes from manifest.json:
        // input:    [2, 3, 4]
        // weight:   [4]
        // expected: [2, 3, 4]
        // epsilon:  [1]

        constexpr std::size_t batch_size = 2;
        constexpr std::size_t sequence_length = 3;
        constexpr std::size_t hidden_size = 4;
        const std::uint32_t token_count = checked_u32(checked_multiply(batch_size, sequence_length), "token_count");
        const std::uint32_t expected_count = checked_u32(checked_multiply(token_count, hidden_size), "expected_count");

        // Get fixtures
        llmetal::CpuTensor<float> input_tensor_cpu(
            llmetal::Shape{ batch_size, sequence_length, hidden_size }, 
            read_raw<float>(fixture_directory / "x.f32", expected_count)
        );
        llmetal::CpuTensor<float> weights_tensor_cpu(
            llmetal::Shape{ hidden_size }, 
            read_raw<float>(fixture_directory / "weight.f32", hidden_size)
        );
        llmetal::CpuTensor<float> expected_tensor_cpu(
            llmetal::Shape{ batch_size, sequence_length, hidden_size }, 
            read_raw<float>(fixture_directory / "expected.f32", expected_count)
        );
        float epsilon = read_raw<float>(fixture_directory / "eps.f32", 1)[0];

        // Verify cpu oracle
        llmetal::CpuTensor<float> output_tensor_cpu(
            llmetal::Shape{batch_size, sequence_length, hidden_size}
        );
        llmetal::cpu::rms_norm(input_tensor_cpu, weights_tensor_cpu, output_tensor_cpu, epsilon);

        llmetal::MetalContext context;
        llmetal::RmsNormKernel kernel(context, 32, 1);
        auto input_tensor_gpu = context.upload(input_tensor_cpu);
        auto weights_tensor_gpu = context.upload(weights_tensor_cpu);
        auto output_tensor_gpu = context.allocate<float>(
            llmetal::Shape{batch_size, sequence_length, hidden_size}
        );
        auto job = kernel.submit(
            input_tensor_gpu, weights_tensor_gpu, output_tensor_gpu, epsilon
        );
        job.wait();
        auto output_tensor_gpu_cpu = context.download(output_tensor_gpu);
        
        // Verify cpu oracle.
        bool result = verify_equal(output_tensor_cpu, expected_tensor_cpu) &&
                      verify_equal(output_tensor_gpu_cpu, expected_tensor_cpu);

        if (!result) {
            std::cerr << "RmsNorm test failed." << '\n';
            return 1;
        }

    } catch (const std::exception& error) {
        std::cerr << "RmsNorm test failed: " << error.what() << '\n';
        return 1;
    }

    return 0;
}
