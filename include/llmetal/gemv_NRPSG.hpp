#pragma once

#include <cstddef>
#include <memory>
#include <span>

#include "llmetal/tensor.hpp"
#include "llmetal/metal_job.hpp"

namespace llmetal {

class MetalContext;

// N rows per SIMD group.
class GemvNRPSGKernel {
public:
    GemvNRPSGKernel(MetalContext& context, std::size_t rpsg, std::size_t sgptg);
    ~GemvNRPSGKernel();
    GemvNRPSGKernel(const GemvNRPSGKernel&) = delete;
    GemvNRPSGKernel& operator=(const GemvNRPSGKernel&) = delete;
    GemvNRPSGKernel(GemvNRPSGKernel&&) noexcept;
    GemvNRPSGKernel& operator=(GemvNRPSGKernel&&) noexcept;

    llmetal::MetalJob submit(
        const GpuTensor<float>& matrix, // [rows, cols]
        const GpuTensor<float>& vector, // [cols]
        GpuTensor<float>& output        // [rows]
    );
    // Used for benchmarking
    llmetal::MetalJob submit_repeated(
        std::size_t repeats,
        const GpuTensor<float>& matrix, // [rows, cols]
        const GpuTensor<float>& vector, // [cols]
        GpuTensor<float>& output        // [rows]
    );

    // Before submit(), this is False.
    // After submit(), this is True if the command buffer is still running.
    // Multiple submissions are allowed - this will be False between completion
    // of the previous command buffer and the start of the new one.
    bool in_progress() const noexcept;
private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace llmetal
