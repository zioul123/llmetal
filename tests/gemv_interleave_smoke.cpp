#include <algorithm>
#include <iostream>
#include <exception>
#include <iomanip>

#include <llmetal/gemv_interleaved.hpp>
#include <llmetal/gemv_shape.hpp>
#include <vector>
#include <cmath>

void print_matrix(std::span<const float> matrix, std::size_t cols, std::size_t rows) {
    float max_val = std::fabs(*std::max_element(matrix.begin(), matrix.end()));
    uint max_digits = std::log10(max_val) + 1;
    
    for (std::size_t r = 0; r < rows; ++r) {
        for (std::size_t c = 0; c < cols; ++c) {
            std::cout << std::right << std::setw(max_digits) << matrix[r * cols + c] << ' ';
        }
        std::cout << '\n';
    }
    std::cout << std::endl;
}

bool run_case(llmetal::GemvShape shape) {
    std::vector<float> matrix(shape.cols * shape.rows);
    for (std::size_t i = 0; i < matrix.size(); ++i) {
        matrix[i] = i;
    }
    std::cout << "Matrix:" << std::endl;
    print_matrix(matrix, shape.cols, shape.rows);

    for (std::size_t threads_in_group : { 2, 4 }) {
        std::size_t groups_in_grid = (shape.rows + threads_in_group - 1) / threads_in_group;
        std::cout << "Threads in group: " << threads_in_group << std::endl;
        std::vector interleaved2tprg = llmetal::interleave(matrix, groups_in_grid, threads_in_group, shape);
        // If it's not a matching grid size, just print with original dimensions
        if ((std::size_t)(matrix.size() / groups_in_grid) * groups_in_grid != matrix.size()) {
            print_matrix(interleaved2tprg, shape.cols, shape.rows);
        } else {
            print_matrix(interleaved2tprg, matrix.size() / groups_in_grid, groups_in_grid);
        }
    }
    return true;
}

int main() {
    try {
        // Exercises small work, a larger dispatch, and reuse after shrinking.
        for (const auto shape: {
            llmetal::GemvShape{4, 4},
            llmetal::GemvShape{8, 4},
            llmetal::GemvShape{4, 8}, // Same capacity, different dimensions
            llmetal::GemvShape{8, 8},
            llmetal::GemvShape{3, 3},

        }) {
            if (!run_case(shape)) return 1;
        }
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "Vector-add smoke test failed: " << error.what() << '\n';
        return 1;
    }
}
