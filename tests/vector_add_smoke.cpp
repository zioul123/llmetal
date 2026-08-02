#include <llmetal/metal_context.hpp>
#include <llmetal/vector_add.hpp>

#include <cmath>
#include <cstddef>
#include <exception>
#include <algorithm>
#include <iostream>
#include <random>
#include <vector>

namespace {

    
bool run_case(llmetal::VectorAddKernel& kernel, std::size_t element_count) {
    std::vector<float> lhs(element_count);
    std::vector<float> rhs(element_count);
    std::vector<float> output(element_count);
    
    std::mt19937 rng(element_count);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    std::generate(lhs.begin(), lhs.end(), [&](){ return dist(rng); });
    std::generate(rhs.begin(), rhs.end(), [&](){ return dist(rng); });

    kernel.prepare(element_count);
    kernel.upload(lhs, rhs);
    kernel.run();
    kernel.download(output);

    constexpr float tolerance = 1.0e-6f;
    for (std::size_t index = 0; index < element_count; ++index) {
        const float expected = lhs[index] + rhs[index];
        if (std::fabs(output[index] - expected) > tolerance) {
            std::cerr << "Mismatch at index " << index
                      << ": expected " << expected
                      << ", got " << output[index] << '\n';
            return false;
        }
    }

    return true;
}

} // namespace

int main() {
    try {
        llmetal::MetalContext context;
        llmetal::VectorAddKernel kernel(context);

        // Exercises small work, a larger dispatch, and reuse after shrinking.
        return run_case(kernel, 3) && run_case(kernel, 257) && run_case(kernel, 19)
            ? 0
            : 1;
    } catch (const std::exception& error) {
        std::cerr << "Vector-add smoke test failed: " << error.what() << '\n';
        return 1;
    }
}
