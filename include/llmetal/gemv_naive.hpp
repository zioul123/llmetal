#pragma once

#include <cstddef>
#include <memory>
#include <span>

#include "llmetal/tensor.hpp"
#include "llmetal/metal_job.hpp"

namespace llmetal {

class MetalContext;

class GemvNaiveKernel {
public:
    GemvNaiveKernel(MetalContext& context);
    ~GemvNaiveKernel();
    GemvNaiveKernel(const GemvNaiveKernel&) = delete;
    GemvNaiveKernel& operator=(const GemvNaiveKernel&) = delete;
    GemvNaiveKernel(GemvNaiveKernel&&) noexcept;
    GemvNaiveKernel& operator=(GemvNaiveKernel&&) noexcept;

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
