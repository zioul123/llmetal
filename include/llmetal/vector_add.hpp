#pragma once

#include <cstddef>
#include <span>
#include <memory>

namespace llmetal {

class MetalContext;

class VectorAddKernel {
public:
    VectorAddKernel(MetalContext& context);
    ~VectorAddKernel();

    VectorAddKernel(const VectorAddKernel&) = delete;
    VectorAddKernel& operator=(const VectorAddKernel&) = delete;

    VectorAddKernel(VectorAddKernel&&) noexcept;
    VectorAddKernel& operator=(VectorAddKernel&&) noexcept;

    void prepare(std::size_t element_count);
    void upload(std::span<const float> lhs, std::span<const float> rhs);
    void run();
    void download(std::span<float> output);
    
private:
    class Impl;
    std::unique_ptr<Impl> impl_;
    
    // MTLComputeCommandEncoder
    void encodeAddCommand(void* computeEncoder); 
};

} // namespace llmetal