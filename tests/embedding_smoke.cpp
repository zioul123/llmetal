#include "llmetal/cpu/embedding.hpp"
#include <cstdint>
#include <llmetal/tensor.hpp>
#include "llmetal/embedding.hpp"
#include "llmetal/metal_context.hpp"
#include "llmetal/io/reader.hpp"
#include "llmetal/verification/equality.hpp"

#include <iostream>
#include <vector>

int main() {
    try {
        const std::filesystem::path fixture_directory = "./tests/fixtures/embedding";

        // Shapes from manifest.json:
        // ids:      [2, 3]
        // table:    [4, 5]
        // expected: [2, 3, 5]

        constexpr std::size_t batch_size = 2;
        constexpr std::size_t sequence_length = 3;
        constexpr std::uint32_t vocab_size = 4;
        constexpr std::size_t hidden_size = 5;

        constexpr std::size_t token_count = checked_multiply(batch_size, sequence_length);
        constexpr std::size_t table_count = checked_multiply(vocab_size, hidden_size);
        constexpr std::size_t expected_count = checked_multiply(checked_multiply(batch_size,
                                                                                     sequence_length), 
                                                                hidden_size);

        // Convert the ids from long to uint32
        std::vector<long long> ids_vector_long(
            read_raw<std::int64_t>(fixture_directory / "ids.i64", 
                                   token_count)
        );
        std::vector<std::uint32_t> ids_tensor_int(ids_vector_long.begin(), ids_vector_long.end());

        llmetal::CpuTensor<std::uint32_t> ids_tensor(
            llmetal::Shape{batch_size, sequence_length}, 
            ids_tensor_int
        );

        llmetal::CpuTensor<float> table_tensor(
            llmetal::Shape{vocab_size, hidden_size}, 
            read_raw<std::float_t>(fixture_directory / "table.f32", table_count)
        );

        llmetal::CpuTensor<float> expected_tensor(
            llmetal::Shape{batch_size, sequence_length, hidden_size}, 
            read_raw<std::float_t>(fixture_directory / "expected.f32", expected_count)
        );

        llmetal::CpuTensor<float> output_tensor(
            llmetal::Shape{batch_size, sequence_length, hidden_size}
        );
        llmetal::cpu::embedding(table_tensor, ids_tensor, output_tensor, vocab_size);
        
        llmetal::MetalContext context;
        llmetal::EmbeddingKernel kernel(context, 32, 1);
        auto table_tensor_gpu = context.upload(table_tensor);
        auto ids_tensor_gpu = context.upload(ids_tensor);
        auto output_tensor_gpu = context.allocate<float>(
            llmetal::Shape{batch_size, sequence_length, hidden_size}
        );
        auto job = kernel.submit(
            table_tensor_gpu, ids_tensor_gpu, output_tensor_gpu
        );
        job.wait();
        auto output_tensor_gpu_cpu = context.download(output_tensor_gpu);

        // Verify cpu oracle and gpu result match.
        bool result = verify_equal(output_tensor, expected_tensor) &&
                      verify_equal(output_tensor_gpu_cpu, expected_tensor);
        if (!result) {
            std::cerr << "Embedding test failed." << '\n';
            return 1;
        }

    } catch (const std::exception& error) {
        std::cerr << "Embedding test failed: " << error.what() << '\n';
        return 1;
    }

    return 0;
}
