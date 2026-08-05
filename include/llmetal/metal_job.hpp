#pragma once

#include <chrono>
#include <memory>

namespace llmetal {
 
class MetalJob {
public:
    ~MetalJob();
    MetalJob(const MetalJob&) = delete;
    MetalJob& operator=(const MetalJob&) = delete;
    MetalJob(MetalJob&&) noexcept;
    MetalJob& operator=(MetalJob&&) noexcept;

    void wait(); // Surfaces command-buffer failures
    [[nodiscard]] bool completed() const noexcept;
    [[nodiscard]] std::chrono::nanoseconds gpu_duration() const;
private:
    MetalJob(void* commandBuffer);
    
    class Impl;
    std::unique_ptr<Impl> impl_;

    friend class GemvNaiveKernel;
    friend class GemvMpsKernel;
    friend class GemvNRPSGKernel;
};

} // namespace llmetal
