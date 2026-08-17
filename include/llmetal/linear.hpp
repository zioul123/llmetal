#pragma once
#include <memory>
#include "llmetal/metal_job.hpp"
#include "llmetal/tensor.hpp"
namespace llmetal { class MetalContext;
class LinearKernel {
public:
    explicit LinearKernel(MetalContext&, std::uint32_t tptg, std::uint32_t tpr);
    ~LinearKernel();
    LinearKernel(const LinearKernel&)=delete;
    LinearKernel& operator=(const LinearKernel&)=delete;
    LinearKernel(LinearKernel&&) noexcept;
    LinearKernel& operator=(LinearKernel&&) noexcept;

    // Without bias
    [[nodiscard]] MetalJob submit(const GpuTensor<float>& input,   // [B,S,I]
                                  const GpuTensor<float>& weight,  // [O,I]
                                  GpuTensor<float>& output);       // [B,S,O]
    [[nodiscard]] MetalJob submit_repeated(
                                    std::size_t repeats,
                                    const GpuTensor<float>& input,   // [B,S,I]
                                    const GpuTensor<float>& weight,  // [O,I]
                                    GpuTensor<float>& output);       // [B,S,O]
    // With bias
    [[nodiscard]] MetalJob submit(const GpuTensor<float>& input,                               const GpuTensor<float>& weight,
                                  const GpuTensor<float>& bias,  // [O]
                                  GpuTensor<float>& output);
    [[nodiscard]] MetalJob submit_repeated(
                                    std::size_t repeats,
                                    const GpuTensor<float>& input,                                 const GpuTensor<float>& weight,
                                    const GpuTensor<float>& bias,  // [O]
                                    GpuTensor<float>& output);
    [[nodiscard]] bool in_progress() const noexcept;
private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

}
