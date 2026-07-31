#include <llmetal/metal_context.hpp>

#include <iostream>
#include <exception>

int main() {
    std::cout << "Hello, World!" << std::endl;
    std::cout << "Build: " << LLMETAL_BUILD_TYPE << "\n";

    try {
        print_metal_device();
        return 0;
    } catch(const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
}