#pragma once
#include "llmetal/tensor.hpp"

inline bool verify_equal(llmetal::CpuTensor<float>& actual, llmetal::CpuTensor<float>& expect) {
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
