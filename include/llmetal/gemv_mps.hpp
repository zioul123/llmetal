#pragma once

#include <cstddef>
#include <memory>
#include <span>

#include "llmetal/metal_job.hpp"
#include "llmetal/tensor.hpp"

namespace llmetal {

class MetalContext;

class GemvMpsKernel {
public:
    GemvMpsKernel(MetalContext& context);
    ~GemvMpsKernel();
    GemvMpsKernel(const GemvMpsKernel&) = delete;
    GemvMpsKernel& operator=(const GemvMpsKernel&) = delete;
    GemvMpsKernel(GemvMpsKernel&&) noexcept;
    GemvMpsKernel& operator=(GemvMpsKernel&&) noexcept;

    void prepare(Shape);
    void upload_matrix(std::span<const float> matrix);
    void upload_vector(std::span<const float> vector);
    llmetal::MetalJob submit();
    // Used for benchmarking
    llmetal::MetalJob submit_repeated(std::size_t repeats);
    void download(std::span<float> output);

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
