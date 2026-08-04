#include "llmetal/metal_job.hpp"

#include <chrono>
#include <stdexcept>

#include <CoreFoundation/CFDate.h>
#include <Foundation/Foundation.h>
#include <Metal/Metal.h>

namespace llmetal {

class MetalJob::Impl {
public:
    explicit Impl(void* commandBuffer){}
private:
    id<MTLCommandBuffer> commandBuffer = nil;
friend class MetalJob;
};

MetalJob::MetalJob(void* commandBuffer)
    : impl_(std::make_unique<Impl>(commandBuffer)) {
    NSError* error = nil;
    impl_->commandBuffer = (__bridge id<MTLCommandBuffer>)commandBuffer;
}

MetalJob::~MetalJob() = default;
MetalJob::MetalJob(MetalJob&&) noexcept = default;
MetalJob& MetalJob::operator=(MetalJob&&) noexcept = default;

void MetalJob::wait() {
    // Await results
    [impl_->commandBuffer waitUntilCompleted];

    // Throw error if failed
    if (impl_->commandBuffer.error != nil) { 
        throw std::runtime_error(
            std::string("Command buffer failed: ") 
            + [impl_->commandBuffer.error.localizedDescription UTF8String]
        );
    }
}

[[nodiscard]] bool MetalJob::completed() const noexcept {
    return [impl_->commandBuffer status] == MTLCommandBufferStatusCompleted;
}

[[nodiscard]] std::chrono::nanoseconds MetalJob::gpu_duration() const {
    if (!completed()) { 
        throw std::logic_error("gpu_duration() called on incomplete job");
    }
    CFTimeInterval startTime = [impl_->commandBuffer GPUStartTime];
    CFTimeInterval endTime = [impl_->commandBuffer GPUEndTime];
    return std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::duration<double>(endTime - startTime)
    );
}

} // namespace llmetal
