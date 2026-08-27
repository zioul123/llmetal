#pragma once
#include <memory>
#include "llmetal/metal_job.hpp"
#include "llmetal/tensor.hpp"
namespace llmetal { class MetalContext;
class LinearKernel {
public:
    explicit LinearKernel(MetalContext&,
                          std::uint32_t tptg,
                          std::uint32_t tpr,
                          std::uint32_t rpt = 1,
                          std::uint32_t npt = 1);
    ~LinearKernel();
    LinearKernel(const LinearKernel&)=delete;
    LinearKernel& operator=(const LinearKernel&)=delete;
    LinearKernel(LinearKernel&&) noexcept;
    LinearKernel& operator=(LinearKernel&&) noexcept;

    // Without bias
    [[nodiscard]] MetalJob submit(const GpuTensor<float>& input,   // [B,S,I]
                                  const GpuTensor<float>& weight,  // [O,I]
                                  const GpuTensor<float>* bias,    // [O], nullptr = no bias
                                  GpuTensor<float>& output);       // [B,S,O]
    [[nodiscard]] MetalJob submit_repeated(
                                    std::size_t repeats,
                                    const GpuTensor<float>& input,   // [B,S,I]
                                    const GpuTensor<float>& weight,  // [O,I]
                                    const GpuTensor<float>* bias,    // [O], nullptr = no bias
                                    GpuTensor<float>& output);       // [B,S,O]
    [[nodiscard]] bool in_progress() const noexcept;
private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

}
