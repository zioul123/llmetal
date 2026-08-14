#pragma once

#include "llmetal/tensor.hpp"
#include "llmetal/metal_job.hpp"

namespace llmetal {

class MetalContext;

class RoPEKernel {
public:
    // tptg: threads per thread group, tpr: threads per row
    RoPEKernel(MetalContext& context, std::uint32_t tptg, std::uint32_t tpr);
    ~RoPEKernel();
    RoPEKernel(const RoPEKernel&) = delete;
    RoPEKernel& operator=(const RoPEKernel&) = delete;
    RoPEKernel(RoPEKernel&&) noexcept;
    RoPEKernel& operator=(RoPEKernel&&) noexcept;

    llmetal::MetalJob submit(
        const GpuTensor<float>& input,      // [batch, seq_length, num_heads, head_dim]
        GpuTensor<float>& output,           // [batch, seq_length, num_heads, head_dim]
        const GpuTensor<float>& cos_and_sin // [max_seq_length, rotary_dim / 2, 2 (cos_and_sin)]
    );
    llmetal::MetalJob submit_repeated(
        std::size_t repeats,
        const GpuTensor<float>& input,      // [batch, seq_length, num_heads, head_dim]
        GpuTensor<float>& output,           // [batch, seq_length, num_heads, head_dim]
        const GpuTensor<float>& cos_and_sin // [max_seq_length, rotary_dim / 2, 2 (cos_and_sin)]
    );

    bool in_progress() const noexcept;
private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace llmetal
