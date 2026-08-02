#pragma once

#include <cstddef>
#include <memory>
#include <span>

namespace llmetal {

class MetalContext;

struct GemvShape {
    std::size_t rows; // Output elements
    std::size_t cols; // Input elements
};

class GemvKernel {
public:
    GemvKernel(MetalContext& context);
    ~GemvKernel();
    GemvKernel(const GemvKernel&) = delete;
    GemvKernel& operator=(const GemvKernel&) = delete;
    GemvKernel(GemvKernel&&) noexcept;
    GemvKernel& operator=(GemvKernel&&) noexcept;

    void prepare(GemvShape);
    void upload_matrix(std::span<const float> matrix);
    void upload_vector(std::span<const float> vector);
    void run();
    void download(std::span<float> output);

private:
    class Impl;
    std::unique_ptr<Impl> impl_;

    // MTLComputeCommandEncoder
    void encodeAddCommand(void* computeEncoder); 
};

} // namespace llmetal