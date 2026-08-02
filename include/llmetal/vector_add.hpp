#pragma once

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

    void run(
        std::span<const float> lhs,
        std::span<const float> rhs,
        std::span<float> output   
    );
    
private:
    class Impl;
    std::unique_ptr<Impl> impl_;
    
    void encodeAddCommand(
        // MTLComputeCommandEncoder
        void* computeEncoder,
         // MTLBuffer
        void* _bufferLhs,
        // MTLBuffer
        void* _bufferRhs,
        // MTLBuffer
        void* _bufferOutput,
        unsigned int elementCount
    ); 
};

} // namespace llmetal