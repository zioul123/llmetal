#include "llmetal/cpu/embedding.hpp"
#include <llmetal/tensor.hpp>
#include "llmetal/embedding.hpp"
#include "llmetal/metal_context.hpp"

#include <fstream>
#include <iostream>
#include <vector>

template <typename T>
std::vector<T> read_raw(
    const std::filesystem::path& path,
    std::size_t expected_elements
) {
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file) {
        throw std::runtime_error("Could not open: " + path.string());
    }

    const std::streamsize byte_count = file.tellg();
    const std::size_t expected_bytes = expected_elements * sizeof(T);

    if (byte_count < 0 ||
        static_cast<std::size_t>(byte_count) != expected_bytes) {
        throw std::runtime_error(
            "Unexpected file size for " + path.string() +
            ": expected " + std::to_string(expected_bytes) +
            " bytes, got " + std::to_string(byte_count)
        );
    }

    file.seekg(0, std::ios::beg);

    std::vector<T> values(expected_elements);
    if (!file.read(
            reinterpret_cast<char*>(values.data()),
            static_cast<std::streamsize>(expected_bytes)
        )) {
        throw std::runtime_error("Could not read: " + path.string());
    }

    return values;
}

bool verify_equal(llmetal::CpuTensor<float>& actual, llmetal::CpuTensor<float>& expect) {
    std::size_t expected_count = expect.span().size();
    std::size_t actual_count = actual.span().size();
    if (expected_count != actual_count) {
        std::cerr << "Mismatch in tensor size: expected " << expect.shape() << ", got " << actual.shape() << '\n';
        return false;
    }

    constexpr float tolerance = 1.0e-6f;
    for (std::size_t index = 0; index < expected_count; ++index) {
        if (std::fabs(expect[index] - actual[index]) > tolerance) {
            std::cerr << "Mismatch at index " << index
                      << ": expected " << expect[index] << '\n';
            
            // Print out inputs and outputs to help debugging.
            std::cout << "actual " << actual.shape() << ":\n"; 
            std::cout << actual << std::endl;
    
            std::cout << "expect "  << expect.shape() << ":\n" << std::right;
            std::cout << expect << std::endl;
            return false;
        }
    }
    return true;
}

int main() {
    try {
        const std::filesystem::path fixture_directory = "./tests/fixtures/embedding";

        // Shapes from manifest.json:
        // ids:      [2, 3]
        // table:    [4, 5]
        // expected: [2, 3, 5]

        constexpr std::size_t batch_size = 2;
        constexpr std::size_t sequence_length = 3;
        constexpr std::size_t vocab_size = 4;
        constexpr std::size_t hidden_size = 5;

        constexpr std::size_t token_count = batch_size * sequence_length;
        constexpr std::size_t table_count = vocab_size * hidden_size;
        constexpr std::size_t expected_count = batch_size * sequence_length * hidden_size;

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
        llmetal::cpu::embedding(table_tensor, ids_tensor, output_tensor);
        
        llmetal::MetalContext context;
        llmetal::EmbeddingKernel kernel(context);
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
