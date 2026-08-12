#include <cstdint>
#include <llmetal/tensor.hpp>
#include "llmetal/cpu/rope.hpp"
#include "llmetal/metal_context.hpp"
#include "llmetal/verification/equality.hpp"
#include "llmetal/io/reader.hpp"

#include <iostream>
#include <vector>


int main() {
    try {
        const std::filesystem::path fixture_directory = "./tests/fixtures/rope";

        // Shapes from manifest.json:
        // input (batch, sequence, heads, dim per head): [1, 3, 2, 4]
        // cos   (sequence, dim per head): [3, 4]
        // sin   (sequence, dim per head): [3, 4]
        // expected: [1, 3, 2, 4]

        constexpr std::size_t batch_size      = 1;
        constexpr std::size_t sequence_length = 3;
        constexpr std::size_t num_heads       = 2;
        constexpr std::size_t dim_per_head    = 4;
        constexpr float theta = 100000.0f;

        std::uint32_t rope_count = checked_u32(
            checked_multiply(sequence_length, dim_per_head),
            "rope_count"
        );
        std::uint32_t expected_count = checked_u32(
            checked_multiply(
                checked_multiply(batch_size, sequence_length),
                checked_multiply(num_heads, dim_per_head)
            ),
            "expected_count"
        );

        llmetal::Shape inputShape{ batch_size, sequence_length, num_heads, dim_per_head };
        llmetal::Shape rope_fixture_shape{ sequence_length, dim_per_head };

        // Get fixtures
        llmetal::CpuTensor<float> input_tensor_cpu(
            inputShape,
            read_raw<float>(fixture_directory / "x.f32", expected_count)
        );
        llmetal::CpuTensor<float> cos(
            rope_fixture_shape,
            read_raw<float>(fixture_directory / "cos.f32", rope_count)
        );
        llmetal::CpuTensor<float> sin(
            rope_fixture_shape,
            read_raw<float>(fixture_directory / "sin.f32", rope_count)
        );
        llmetal::CpuTensor<float> expected_tensor_cpu(
            inputShape,
            read_raw<float>(fixture_directory / "expected.f32", expected_count)
        );

        // Verify cpu oracle
        // llmetal::CpuTensor<float> output_tensor_cpu(
        //     llmetal::Shape{ batch_size, sequence_length, num_heads, dim_per_head }
        // );
        llmetal::Shape rope_packed_shape{ sequence_length, dim_per_head / 2, 2 };

        llmetal::CpuTensor<float> rope_output_tensor_cpu(
            llmetal::Shape{ sequence_length, dim_per_head / 2, 2 }
        );
        llmetal::cpu::rope_cos_and_sin(sequence_length, dim_per_head, theta, rope_output_tensor_cpu);

        // Repack for comparison
        llmetal::CpuTensor<float> rope_cos(rope_fixture_shape);
        llmetal::CpuTensor<float> rope_sin(rope_fixture_shape);
        for (std::uint32_t seq_idx = 0; seq_idx < sequence_length; ++seq_idx) {
            for (std::uint32_t d = 0; d < dim_per_head; ++d) { 
                rope_cos[seq_idx * dim_per_head + d] 
                    = rope_output_tensor_cpu[seq_idx * dim_per_head + (d * 2 % (dim_per_head / 2))];
                rope_sin[seq_idx * dim_per_head + d] 
                    = rope_output_tensor_cpu[seq_idx * dim_per_head + (1 + d * 2 % (dim_per_head / 2))];
            }
        }


        // llmetal::MetalContext context;
        // llmetal::RmsNormKernel kernel(context, 32, 1);
        // auto input_tensor_gpu = context.upload(input_tensor_cpu);
        // auto weights_tensor_gpu = context.upload(weights_tensor_cpu);
        // auto output_tensor_gpu = context.allocate<float>(
        //     llmetal::Shape{batch_size, sequence_length, hidden_size}
        // );
        // auto job = kernel.submit(
        //     input_tensor_gpu, weights_tensor_gpu, output_tensor_gpu, epsilon
        // );
        // job.wait();
        // auto output_tensor_gpu_cpu = context.download(output_tensor_gpu);
        
        // Verify cpu oracle.
        bool result = verify_equal(rope_cos, cos) && verify_equal(rope_sin, sin);
        // bool result = verify_equal(output_tensor_cpu, expected_tensor_cpu); // &&
                    //   verify_equal(output_tensor_gpu_cpu, expected_tensor_cpu);

        if (!result) {
            std::cerr << "Rope test failed." << '\n';
            return 1;
        }

    } catch (const std::exception& error) {
        std::cerr << "Rope test failed: " << error.what() << '\n';
        return 1;
    }

    return 0;
}
