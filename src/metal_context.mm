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

} // namespace llmetal
