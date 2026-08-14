#include <cstdint>
#include <llmetal/tensor.hpp>
#include "llmetal/cpu/rope.hpp"
#include "llmetal/metal_context.hpp"
#include "llmetal/rope.hpp"
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

        constexpr std::size_t batch_size = 1;
        constexpr std::size_t seq_length = 3;
        constexpr std::size_t num_heads  = 2;
        constexpr std::size_t head_dim   = 4;
        constexpr float theta = 100000.0f;

        std::uint32_t rope_count = checked_u32(
            checked_multiply(seq_length, head_dim),
            "rope_count"
        );
        std::uint32_t expected_count = checked_u32(
            checked_multiply(
                checked_multiply(batch_size, seq_length),
                checked_multiply(num_heads, head_dim)
            ),
            "expected_count"
        );

        llmetal::Shape inputShape{ batch_size, seq_length, num_heads, head_dim };
        llmetal::Shape rope_fixture_shape{ seq_length, head_dim };

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

        // Verify cpu oracle - cos and sin precomputation generation
        llmetal::Shape rope_packed_shape{ seq_length, head_dim / 2, 2 };
        
        llmetal::CpuTensor<float> rope_cos_and_sin_cpu(
            llmetal::Shape{ seq_length, head_dim / 2, 2 }
        );
        llmetal::cpu::rope_cos_and_sin(seq_length, head_dim, theta, rope_cos_and_sin_cpu);
        
        // Repack for comparison
        llmetal::CpuTensor<float> rope_cos(rope_fixture_shape);
        llmetal::CpuTensor<float> rope_sin(rope_fixture_shape);
        std::uint32_t pair_count = head_dim / 2;
        for (std::uint32_t seq_idx = 0; seq_idx < seq_length; ++seq_idx) {
            for (std::uint32_t d = 0; d < head_dim; ++d) { 
                std::uint32_t pair_idx = d % pair_count;
                std::uint32_t cos_and_sin_idx = seq_idx * pair_count * 2 + pair_idx * 2;
                rope_cos[seq_idx * head_dim + d] 
                    = rope_cos_and_sin_cpu[cos_and_sin_idx];
                rope_sin[seq_idx * head_dim + d] 
                    = rope_cos_and_sin_cpu[cos_and_sin_idx + 1];
            }
        }

        // Verify cpu oracle - rope application
        llmetal::CpuTensor<float> output_tensor_cpu(
            llmetal::Shape{ batch_size, seq_length, num_heads, head_dim }
        );
        llmetal::cpu::rotate_half(input_tensor_cpu, output_tensor_cpu, rope_cos_and_sin_cpu);

        // Verify metal
        llmetal::MetalContext context;
        llmetal::RoPEKernel kernel(context, 32, 1);
        auto input_tensor_gpu = context.upload(input_tensor_cpu);
        auto cos_and_sin_gpu = context.upload(rope_cos_and_sin_cpu);
        auto output_tensor_gpu = context.allocate<float>(inputShape);
        auto job = kernel.submit(
            input_tensor_gpu, output_tensor_gpu, cos_and_sin_gpu
        );
        job.wait();
        auto output_tensor_gpu_cpu = context.download(output_tensor_gpu);
        
        // Verify cpu oracle.
        bool result = verify_equal(rope_cos, cos) && verify_equal(rope_sin, sin) &&
                      verify_equal(output_tensor_cpu, expected_tensor_cpu) &&
                      verify_equal(output_tensor_gpu_cpu, expected_tensor_cpu);
        
        if (!result) {
            std::cerr << "Rope test failed." << '\n';
            return 1;
        }

        llmetal::cpu::rotate_half_in_place(input_tensor_cpu, rope_cos_and_sin_cpu);
        job = kernel.submit(
            input_tensor_gpu, input_tensor_gpu, cos_and_sin_gpu
        );
        job.wait();
        auto output_tensor_gpu_cpu_in_place = context.download(input_tensor_gpu);

        result = verify_equal(input_tensor_cpu, expected_tensor_cpu) &&
                verify_equal(output_tensor_gpu_cpu_in_place, expected_tensor_cpu);
        if (!result) {
            std::cerr << "Rope in-place test failed." << '\n';
            return 1;
        }

    } catch (const std::exception& error) {
        std::cerr << "Rope test failed: " << error.what() << '\n';
        return 1;
    }

    return 0;
}
