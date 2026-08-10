#pragma once

#include "llmetal/tensor.hpp"
#include "llmetal/metal_job.hpp"

namespace llmetal {

class MetalContext;

class RmsNormKernel {
public:
    // tptg: threads per thread group, tpr: threads per row
    RmsNormKernel(MetalContext& context, std::uint32_t tptg, std::uint32_t tpr);
    ~RmsNormKernel();
    RmsNormKernel(const RmsNormKernel&) = delete;
    RmsNormKernel& operator=(const RmsNormKernel&) = delete;
    RmsNormKernel(RmsNormKernel&&) noexcept;
    RmsNormKernel& operator=(RmsNormKernel&&) noexcept;

    llmetal::MetalJob submit(
        const GpuTensor<float>& input,   // [batch_size, sequence_length, hidden]
        const GpuTensor<float>& weights, // [hidden]
        GpuTensor<float>& output,        // [batch_size, sequence_length, hidden]
        const float epsilon
    );
    llmetal::MetalJob submit_repeated(
        std::size_t repeats,
        const GpuTensor<float>& input,   // [batch_size, sequence_length, hidden]
        const GpuTensor<float>& weights, // [hidden]
        GpuTensor<float>& output,        // [batch_size, sequence_length, hidden]
        const float epsilon
    );

    bool in_progress() const noexcept;
private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace llmetal
