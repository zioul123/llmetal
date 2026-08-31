#include <cstdint>
#include <llmetal/tensor.hpp>
// #include "llmetal/metal_context.hpp"
#include "llmetal/cpu/gqa.hpp"
#include "llmetal/verification/equality.hpp"
#include "llmetal/io/reader.hpp"

#include <iostream>
#include <string>

using uint = std::uint32_t;

int main() {
    try {
        const std::filesystem::path fixture_directory = "./tests/fixtures/gqa_prefill";

        // q:        [2, 7, 6, 4] (B, S, NQ, D)
        // k:        [2, 7, 3, 4] (B, S, NKV, D)
        // v:        [2, 7, 3, 4] (B, S, NKV, D)
        // expected: [2, 7, 6, 4] (B, S, NQ, D)
        
        constexpr uint B = 2;
        constexpr uint S = 7;
        constexpr uint NQ = 6;
        constexpr uint NKV = 3;
        constexpr uint D = 4;

        uint q_count = B * S * NQ * D;
        uint kv_count = B * S * NKV * D;
        auto q_shape = llmetal::Shape{ B, S, NQ, D };
        auto kv_shape = llmetal::Shape{ B, S, NKV, D };
        
        // Get fixtures
        llmetal::CpuTensor<float> q_tensor_cpu(q_shape, read_raw<float>(fixture_directory / "q.f32", q_count)
        );
        llmetal::CpuTensor<float> k_tensor_cpu(kv_shape, read_raw<float>(fixture_directory / "k.f32", kv_count));
        llmetal::CpuTensor<float> v_tensor_cpu(kv_shape, read_raw<float>(fixture_directory / "v.f32", kv_count));
        llmetal::CpuTensor<float> expected_tensor_cpu(q_shape, read_raw<float>(fixture_directory / "expected.f32", q_count));
        
        // Verify cpu oracle
        llmetal::CpuTensor<float> output_tensor_cpu(q_shape);
        llmetal::cpu::gqaPrefill(q_tensor_cpu, k_tensor_cpu, v_tensor_cpu, output_tensor_cpu);

        // Verify cpu oracle.
        bool result = verify_equal(output_tensor_cpu, expected_tensor_cpu);

        if (!result) {
            std::cerr << "gqa prefill test failed - cpu oracle not matching." << '\n';
            return 1;
        }

        // Verify cpu flash oracle with different tiles
        for (auto BQ : { 1, 2, 4, 7, 8}) {
            for (auto BK : { 1, 2, 4, 7, 8}) {

                llmetal::CpuTensor<float> output_tensor_cpu_flash(q_shape);
                llmetal::cpu::gqaPrefillFlash(q_tensor_cpu,
                                              k_tensor_cpu,
                                              v_tensor_cpu,
                                              output_tensor_cpu_flash,
                                              BQ, BK);
        
                // Verify cpu oracle.
                result = verify_equal(output_tensor_cpu_flash, expected_tensor_cpu);
                    
                if (!result) {
                    std::cerr << "gqa flash prefill test failed - cpu oracle not matching for BQ, BK " 
                              << std::to_string(BQ) << " "
                              << std::to_string(BK) << '\n';
                    return 1;
                }
            }
        }

        // llmetal::MetalContext context;
        // llmetal::SoftmaxKernel kernel(context, 32, 1, 1);
        // auto logits_tensor_gpu = context.upload(logits_tensor_cpu);
        // auto valid_tensor_gpu = context.upload(valid_tensor_cpu);
        // auto output_tensor_gpu = context.allocate<float>(
        //     llmetal::Shape{ n_input, max_seq }
        // );
        
        // // // Verify with bias
        // auto job = kernel.submit(
        //     logits_tensor_gpu, valid_tensor_gpu, output_tensor_gpu
        // );
        // job.wait();
        // auto output_tensor_gpu_cpu = context.download(output_tensor_gpu);
        // result = verify_equal(output_tensor_gpu_cpu, expected_tensor_cpu);

        // if (!result) {
        //     std::cerr << "Softmax test failed." << '\n';
        //     return 1;
        // }
    } catch (const std::exception& error) {
        std::cerr << "gqa test failed: " << error.what() << '\n';
        return 1;
    }

    return 0;
}
