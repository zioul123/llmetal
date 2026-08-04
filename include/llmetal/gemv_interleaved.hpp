#pragma once

#include <cstddef>
#include <memory>
#include <span>

#include "llmetal/gemv_shape.hpp"
#include "llmetal/metal_job.hpp"

namespace llmetal {

class MetalContext;

class GemvInterleavedKernel {
public:
    GemvInterleavedKernel(MetalContext& context);
    ~GemvInterleavedKernel();
    GemvInterleavedKernel(const GemvInterleavedKernel&) = delete;
    GemvInterleavedKernel& operator=(const GemvInterleavedKernel&) = delete;
    GemvInterleavedKernel(GemvInterleavedKernel&&) noexcept;
    GemvInterleavedKernel& operator=(GemvInterleavedKernel&&) noexcept;

    void prepare(GemvShape);
    void upload_matrix(std::span<const float> matrix, bool isAlreadyInterleaved = false);
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

std::vector<float> interleave(
    std::span<const float> matrix,
    std::size_t row_groups,
    std::size_t threads_per_row_group,
    GemvShape shape
);

} // namespace llmetal
