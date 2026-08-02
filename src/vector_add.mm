#include <cstddef>
#include <cstring>
#include <format>
#include <algorithm>
#include <iostream>
#include <stdexcept>

#include <llmetal/metal_context.hpp>
#include <llmetal/vector_add.hpp>

#include <Foundation/Foundation.h>
#include <Metal/Metal.h>

namespace llmetal {

class VectorAddKernel::Impl {
public:
    explicit Impl(MetalContext& context): metalContext(context) {}
    MetalContext& metalContext;
private:
    id<MTLComputePipelineState> pipeline_state = nil;
    id<MTLBuffer> bufferLhs = nil;
    id<MTLBuffer> bufferRhs = nil;
    id<MTLBuffer> bufferOut = nil;
    std::size_t capacity_elements = 0;
    std::size_t element_count = 0;
friend class VectorAddKernel;
};

VectorAddKernel::VectorAddKernel(MetalContext& context)
    : impl_(std::make_unique<Impl>(context)) {
    NSError* error = nil;
    id<MTLDevice> device = (__bridge id<MTLDevice>)context.device_handle();

    // We need to manually get the library from the shaders folder
    NSURL* exeURL = [[NSBundle mainBundle] executableURL];
    NSURL* libURL = [
        [exeURL URLByDeletingLastPathComponent] URLByAppendingPathComponent:@"shaders/vector_add.metallib"
    ];  
    id<MTLLibrary> library = [device newLibraryWithURL:libURL error:&error];
    if (library == nil) {
        throw std::runtime_error(
            std::string("Library error: ") + [error.localizedDescription UTF8String]
        );
    }

    // From `shaders/vector_add.metal`
    id<MTLFunction> function = [library newFunctionWithName:@"my_add_arrays"];
    if (function == nil) throw std::runtime_error("Function not found");

    // Create the pipeline state
    impl_->pipeline_state = [device newComputePipelineStateWithFunction: function error:&error];
    if (impl_->pipeline_state == nil) { 
        throw std::runtime_error(
            std::string("Failed to create pipeline state: ") 
            + [error.localizedDescription UTF8String]
        );
    }
}

VectorAddKernel::~VectorAddKernel() = default;
VectorAddKernel::VectorAddKernel(VectorAddKernel&&) noexcept = default;
VectorAddKernel& VectorAddKernel::operator=(VectorAddKernel&&) noexcept = default;

void VectorAddKernel::prepare(std::size_t element_count) {
    if (element_count <= 0) {
        throw std::runtime_error("Invalid element count");
    }

    // Our buffers already have sufficient capacity - just update the element count
    if (element_count <= impl_->capacity_elements) {
        impl_->element_count = element_count;
        return;
    }
    
    id<MTLDevice> device = (__bridge id<MTLDevice>)impl_->metalContext.device_handle();
    id<MTLBuffer> bufferLhs = [device newBufferWithLength:element_count * sizeof(float) options:MTLResourceStorageModeShared];
    id<MTLBuffer> bufferRhs = [device newBufferWithLength:element_count * sizeof(float) options:MTLResourceStorageModeShared];
    id<MTLBuffer> bufferOut = [device newBufferWithLength:element_count * sizeof(float) options:MTLResourceStorageModeShared];
    
    if (bufferLhs == nil || bufferRhs == nil || bufferOut == nil) {
        throw std::runtime_error("Failed to create buffers");
    }

    impl_->bufferLhs = bufferLhs;
    impl_->bufferRhs = bufferRhs;
    impl_->bufferOut = bufferOut;
    impl_->element_count     = element_count;
    impl_->capacity_elements = element_count;

}

void VectorAddKernel::upload(
    std::span<const float> lhs, 
    std::span<const float> rhs
) {
    if (lhs.size() != impl_->element_count || rhs.size() != impl_->element_count) { 
        throw std::runtime_error(
            "Invalid input sizes. Expected: " 
            + std::to_string(impl_->element_count) 
            + ", got: " + std::to_string(lhs.size()) 
            + " and " + std::to_string(rhs.size())
        );
    }
    std::memcpy([impl_->bufferLhs contents], lhs.data(), lhs.size() * sizeof(float));
    std::memcpy([impl_->bufferRhs contents], rhs.data(), rhs.size() * sizeof(float));
}

void VectorAddKernel::run() {
    if (impl_->element_count <= 0) throw std::runtime_error("Invalid element count");

    id<MTLCommandQueue> commandQueue = (__bridge id<MTLCommandQueue>)impl_->metalContext.command_queue_handle();

    // Create command buffer and encoder
    id<MTLCommandBuffer> commandBuffer = [commandQueue commandBuffer];
    if (commandBuffer == nil) throw std::runtime_error("Could not create command buffer");
    id<MTLComputeCommandEncoder> computeEncoder = [commandBuffer computeCommandEncoder];
    if (computeEncoder == nil) throw std::runtime_error("Could not create compute command encoder");

    // Encode command
    encodeAddCommand((__bridge void*) computeEncoder);

    // Execute command
    [commandBuffer commit];

    // Await results
    [commandBuffer waitUntilCompleted];

    // Throw error if failed
    if (commandBuffer.error != nil) { 
        throw std::runtime_error(
            std::string("Command buffer failed: ") 
            + [commandBuffer.error.localizedDescription UTF8String]
        );
    }
}

void VectorAddKernel::download(std::span<float> output) {
    if (output.size() < impl_->element_count) { 
        throw std::runtime_error(
            "Invalid output size. Expected: " 
            + std::to_string(impl_->element_count) 
            + ", got: " + std::to_string(output.size())
        );
    }
    std::memcpy(output.data(), [impl_->bufferOut contents], impl_->element_count * sizeof(float));
}

void VectorAddKernel::encodeAddCommand(void* _computeEncoder) {
    id<MTLComputeCommandEncoder> computeEncoder = (__bridge id<MTLComputeCommandEncoder>)_computeEncoder;

    // Encode the compute command
    [computeEncoder setComputePipelineState:impl_->pipeline_state];
    [computeEncoder setBuffer:impl_->bufferLhs offset:0 atIndex:0];
    [computeEncoder setBuffer:impl_->bufferRhs offset:0 atIndex:1];
    [computeEncoder setBuffer:impl_->bufferOut offset:0 atIndex:2];

    MTLSize gridSize = MTLSizeMake(impl_->element_count, 1, 1);

    NSUInteger upperBound = impl_->pipeline_state.maxTotalThreadsPerThreadgroup;
    if (upperBound > gridSize.width) {
        upperBound = gridSize.width;
    }
    MTLSize threadGroupSize = MTLSizeMake(upperBound, 1, 1);

    [computeEncoder dispatchThreads:gridSize threadsPerThreadgroup:threadGroupSize];
    [computeEncoder endEncoding];
}

} //namespace llmetal