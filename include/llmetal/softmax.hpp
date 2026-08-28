#pragma once

#include "llmetal/tensor.hpp"
#include "llmetal/metal_job.hpp"

namespace llmetal {

class MetalContext;

class SoftmaxKernel {
public:
    // tptg: threads per thread group, tpr: threads per row
    SoftmaxKernel(MetalContext& context, std::uint32_t tptg, std::uint32_t tpr, std::uint32_t rpt);
    ~SoftmaxKernel();
    SoftmaxKernel(const SoftmaxKernel&) = delete;
    SoftmaxKernel& operator=(const SoftmaxKernel&) = delete;
    SoftmaxKernel(SoftmaxKernel&&) noexcept;
    SoftmaxKernel& operator=(SoftmaxKernel&&) noexcept;

    llmetal::MetalJob submit(
        const GpuTensor<float>& logits,        // [rows,C]
        const GpuTensor<std::uint32_t>& valid, // [rows]
        GpuTensor<float>& probs                // [rows,C]
    );
    llmetal::MetalJob submit_repeated(
        std::size_t repeats,
        const GpuTensor<float>& logits,        // [rows,C]
        const GpuTensor<std::uint32_t>& valid, // [rows]
        GpuTensor<float>& probs                // [rows,C]
    );

    bool in_progress() const noexcept;
private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace llmetal
