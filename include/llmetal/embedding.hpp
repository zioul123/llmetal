


#include "llmetal/tensor.hpp"
#include "llmetal/metal_job.hpp"

namespace llmetal {

class MetalContext;

class EmbeddingKernel {
public:
    EmbeddingKernel(MetalContext& context);
    ~EmbeddingKernel();
    EmbeddingKernel(const EmbeddingKernel&) = delete;
    EmbeddingKernel& operator=(const EmbeddingKernel&) = delete;
    EmbeddingKernel(EmbeddingKernel&&) noexcept;
    EmbeddingKernel& operator=(EmbeddingKernel&&) noexcept;

    llmetal::MetalJob submit(
        const GpuTensor<float>& table,       // [vocab_size, hidden]
        const GpuTensor<std::uint32_t>& ids, // [batch_size, sequence_length]
        const GpuTensor<float>& output       // [batch_size, sequence_length, hidden]
    );
    llmetal::MetalJob submit_repeated(
        std::size_t repeats,
        const GpuTensor<float>& table,       // [vocab_size, hidden]
        const GpuTensor<std::uint32_t>& ids, // [batch_size, sequence_length]
        const GpuTensor<float>& output       // [batch_size, sequence_length, hidden]
    );

    bool in_progress() const noexcept;
private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace llmetal
