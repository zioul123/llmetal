#include <cstdint>
#include <llmetal/tensor.hpp>
#include "llmetal/metal_context.hpp"
#include "llmetal/cpu/softmax.hpp"
#include "llmetal/verification/equality.hpp"
#include "llmetal/io/reader.hpp"

#include <iostream>


int main() {
    try {
        const std::filesystem::path fixture_directory = "./tests/fixtures/softmax";

        // Shapes from manifest.json:
        // logits:   [3, 5]
        // valid:    [3]
        // expected: [3, 5]
        
        constexpr std::size_t n_input = 3;
        constexpr std::size_t max_seq = 5;
        const std::uint32_t logits_count = checked_u32(checked_multiply(n_input, max_seq), "logits_count");

        // Get fixtures
        llmetal::CpuTensor<float> logits_tensor_cpu(
            llmetal::Shape{ n_input, max_seq }, 
            read_raw<float>(fixture_directory / "logits.f32", logits_count)
        );

        std::vector<long long> valid_tensor_long(
            read_raw<long long>(fixture_directory / "valid.i64", n_input)
        );
        std::vector<std::uint32_t> valid_tensor_int(valid_tensor_long.begin(), valid_tensor_long.end());
        llmetal::CpuTensor<std::uint32_t> valid_tensor_cpu(
            llmetal::Shape{ n_input},
            valid_tensor_int
        );
        
        llmetal::CpuTensor<float> expected_tensor_cpu(
            llmetal::Shape{ n_input, max_seq }, 
            read_raw<float>(fixture_directory / "expected.f32", logits_count)
        );
        
        // Verify cpu oracle
        llmetal::CpuTensor<float> output_tensor_cpu(llmetal::Shape{ n_input, max_seq });
        llmetal::cpu::softmax(logits_tensor_cpu, valid_tensor_cpu, output_tensor_cpu);

        // Verify cpu oracle.
        bool result = verify_equal(output_tensor_cpu, expected_tensor_cpu);

        if (!result) {
            std::cerr << "Linear test failed - cpu oracle not matching." << '\n';
            return 1;
        }

    } catch (const std::exception& error) {
        std::cerr << "Softmax test failed: " << error.what() << '\n';
        return 1;
    }

    return 0;
}
