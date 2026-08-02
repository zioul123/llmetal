#include <llmetal/metal_context.hpp>
#include "llmetal/vector_add.hpp"

#include <iostream>
#include <exception>
#include <algorithm>
#include <vector>
#include <random>

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
        llmetal::MetalContext context;
        std::cout << "Metal device: " << context.device_name() << std::endl;

        llmetal::VectorAddKernel kernel = llmetal::VectorAddKernel(context);
        
        // Create CPU buffers
        std::vector<float> lhs(3);
        std::vector<float> rhs(3);
        std::vector<float> result(lhs.size());
        
        // Fill CPU buffers with random numbers
        std::mt19937 rng(42);
        std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
        std::generate(lhs.begin(), lhs.end(), [&](){ return dist(rng); });
        std::generate(rhs.begin(), rhs.end(), [&](){ return dist(rng); });
        
        // Prepare and run kernel
        kernel.prepare(lhs.size());
        kernel.upload(lhs, rhs);
        kernel.run();
        kernel.download(result);

        // Print results
        std::cout << "Input LHS: ";
        for (unsigned long index = 0; index < lhs.size(); index++) {
            std::cout << lhs[index] << " ";
        }
        std::cout << std::endl << "Input RHS: ";
        for (unsigned long index = 0; index < rhs.size(); index++) {
            std::cout << rhs[index] << " ";
        }
        std::cout << std::endl << "Output: ";
        for (unsigned long index = 0; index < lhs.size(); index++) {
            std::cout << result[index] << " ";
        }
        std::cout << std::endl;


    } catch(const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}