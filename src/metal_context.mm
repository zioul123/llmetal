#include "llmetal/tensor.hpp"
#include <llmetal/metal_context.hpp>

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <iostream>
#include <cstdint>
#include <stdexcept>

namespace llmetal {

class MetalContext::Impl {
public:
    id<MTLDevice> device = nil;
    id<MTLCommandQueue> command_queue = nil;
};

MetalContext::MetalContext()
    : impl_(std::make_unique<Impl>()) {
    impl_->device = MTLCreateSystemDefaultDevice();

    if (impl_->device == nil) {
        throw std::runtime_error("Metal is unavailable: no default Metal device was found");
    }

    impl_->command_queue = [impl_->device newCommandQueue];

    if (impl_->command_queue == nil) {
        throw std::runtime_error("Failed to create Metal command queue");
    }
}

MetalContext::~MetalContext() = default;

std::string MetalContext::device_name() const {
    const char* device_name = impl_->device.name.UTF8String;
    return device_name != nullptr ? device_name : "<unknown>";
}

void* MetalContext::device_handle() const {
    return (__bridge void*)impl_->device;
}
void* MetalContext::command_queue_handle() const {
    return (__bridge void*)impl_->command_queue;
}

template<typename T>
[[nodiscard]] GpuTensor<T> MetalContext::allocate(Shape shape) const {
    const std::size_t count = shape.numel();
    
    if (count > std::numeric_limits<std::size_t>::max() / sizeof(T)) {
        throw std::runtime_error("GPU Tensor bytes exceeds maximum");
    }

    std::shared_ptr<detail::GpuStorage> storage = detail::make_gpu_storage(
        device_handle(), count * sizeof(T)
    );

    return GpuTensor<T>(std::move(shape), std::move(storage));
}

template<typename T>
[[nodiscard]] GpuTensor<T> MetalContext::upload(const CpuTensor<T>& source) const {
    GpuTensor<T> destination = allocate<T>(source.shape());
    detail::copy_to_storage(destination.storage_, 
                            source.data(), 
                            source.byte_size(), 
                            destination.byte_offset_);
    return destination;
}

template<typename T>
[[nodiscard]] CpuTensor<T> MetalContext::download(const GpuTensor<T>& source) const {
    CpuTensor<T> destination(source.shape());
    detail::copy_from_storage(destination.data(), 
                              source.storage_, 
                              source.byte_size(), 
                              source.byte_offset_);
    return destination;
}

template GpuTensor<float> MetalContext::allocate<float>(Shape) const;
template GpuTensor<std::uint32_t> MetalContext::allocate<std::uint32_t>(Shape) const;

template GpuTensor<float> MetalContext::upload<float>(const CpuTensor<float>&) const;
template GpuTensor<std::uint32_t> MetalContext::upload<std::uint32_t>(
    const CpuTensor<std::uint32_t>&) const;

template CpuTensor<float> MetalContext::download<float>(const GpuTensor<float>&) const;
template CpuTensor<std::uint32_t> MetalContext::download<std::uint32_t>(
    const GpuTensor<std::uint32_t>&) const;

} // namespace llmetal
