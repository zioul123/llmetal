#include <llmetal/metal_context.hpp>
#include "llmetal/vector_add.hpp"

#include <iostream>
#include <exception>

void add_arrays(
    const float* inA,
    const float* inB,
    float* result, 
    int length
) {
    for (int index = 0; index < length ; index++) {
        result[index] = inA[index] + inB[index];
    }
}

int main() {
    std::cout << "LLMetal" << std::endl;
    std::cout << "Build: " << LLMETAL_BUILD_TYPE << "\n";
    try {
        llmetal::MetalContext context = llmetal::MetalContext();
        std::cout << "Metal device: " << context.device_name() << std::endl;

        llmetal::VectorAddKernel kernel = llmetal::VectorAddKernel(context);
        float lhs[] = {1.0f, 2.0f, 3.0f};
        float rhs[] = {4.0f, 5.0f, 6.0f};
        float result[3];
        kernel.run(lhs, rhs, result);

    } catch(const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}