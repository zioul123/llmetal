#include <iostream>
#include <stdexcept>
#include <cstddef>
#include <cstring>
#include <string>
#include <algorithm>

#include <llmetal/metal_context.hpp>
#include <llmetal/gemv.hpp>

#include <Foundation/Foundation.h>
#include <Metal/Metal.h>

namespace llmetal {

class GemvKernel::Impl {
public:
    explicit Impl(MetalContext& context): metalContext(context) {}
    MetalContext& metalContext;
private:
    id<MTLComputePipelineState> pipeline_state = nil;
    id<MTLBuffer> bufferMatrix = nil;
    id<MTLBuffer> bufferVector = nil;
    id<MTLBuffer> bufferOutput = nil;
    std::size_t capacity_elements = 0;
    GemvShape gemvShape = {0, 0};
friend class GemvKernel;
};

GemvKernel::GemvKernel(MetalContext& context)
    : impl_(std::make_unique<Impl>(context)) {
    NSError* error = nil;
    id<MTLDevice> device = (__bridge id<MTLDevice>)context.device_handle();

    // We need to manually get the library from the shaders folder
    NSURL* exeURL = [[NSBundle mainBundle] executableURL];
    NSURL* libURL = [
        [exeURL URLByDeletingLastPathComponent] URLByAppendingPathComponent:@"shaders/gemv.metallib"
    ];  
    id<MTLLibrary> library = [device newLibraryWithURL:libURL error:&error];
    if (library == nil) {
        throw std::runtime_error(
            std::string("Library error: ") + [error.localizedDescription UTF8String]
        );
    }

    // From `shaders/gemv.metal`
    id<MTLFunction> function = [library newFunctionWithName:@"my_gemv"];
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

GemvKernel::~GemvKernel() = default;
GemvKernel::GemvKernel(GemvKernel&&) noexcept = default;
GemvKernel& GemvKernel::operator=(GemvKernel&&) noexcept = default;

void GemvKernel::prepare(GemvShape shape) {
    if (shape.cols <= 0 || shape.rows <= 0) { 
        throw std::runtime_error("Invalid gemv shape");
    }
    
    std::size_t element_count = shape.cols * shape.rows;

    // Our buffers already have sufficient capacity - just update the shape
    if (element_count <= impl_->capacity_elements) {
        impl_->gemvShape = shape;
        return;
    }

    id<MTLDevice> device = (__bridge id<MTLDevice>)impl_->metalContext.device_handle();
    id<MTLBuffer> bufferMatrix = [device newBufferWithLength: element_count * sizeof(float) options: MTLResourceStorageModeShared];
    id<MTLBuffer> bufferVector = [device newBufferWithLength: shape.cols * sizeof(float) options: MTLResourceStorageModeShared];
    id<MTLBuffer> bufferOutput = [device newBufferWithLength: shape.rows * sizeof(float) options: MTLResourceStorageModeShared];

    if (bufferMatrix == nil || bufferVector == nil || bufferOutput == nil) {
        throw std::runtime_error("Failed to create buffers");
    }

    impl_->bufferMatrix = bufferMatrix;
    impl_->bufferVector = bufferVector;
    impl_->bufferOutput = bufferOutput;
    impl_->gemvShape = shape;
    impl_->capacity_elements = element_count;
}

void GemvKernel::upload_matrix(std::span<const float> matrix) {
    if (matrix.size() != impl_->gemvShape.cols * impl_->gemvShape.rows) {
        throw std::runtime_error(
            "Invalid matrix size. Expected: "
            + std::to_string(impl_->gemvShape.cols * impl_->gemvShape.rows)
            + ", got: " + std::to_string(matrix.size())
        );
    }
    std::memcpy([impl_->bufferMatrix contents], matrix.data(), matrix.size() * sizeof(float));
}

void GemvKernel::upload_vector(std::span<const float> vector) {
    if (vector.size() != impl_->gemvShape.cols) {
        throw std::runtime_error(
            "Invalid vector size. Expected: "
            + std::to_string(impl_->gemvShape.cols)
            + ", got: " + std::to_string(vector.size())
        );
    }
    std::memcpy([impl_->bufferVector contents], vector.data(), vector.size() * sizeof(float));
}

void GemvKernel::run() {
    if (impl_->gemvShape.cols <= 0 || impl_->gemvShape.rows <= 0) throw std::runtime_error("Invalid gemv shape");

    id<MTLCommandQueue> commandQueue = (__bridge id<MTLCommandQueue>)impl_->metalContext.command_queue_handle();

    // Create command buffer and encoder
    id<MTLCommandBuffer> commandBuffer = [commandQueue commandBuffer];
    if (commandBuffer == nil) throw std::runtime_error("Could not create command buffer");
    id<MTLComputeCommandEncoder> computeEncoder = [commandBuffer computeCommandEncoder];
    if (computeEncoder == nil) throw std::runtime_error("Could not create compute command encoder");

    // Encode command
    [computeEncoder setComputePipelineState:impl_->pipeline_state];
    [computeEncoder setBuffer:impl_->bufferMatrix offset:0 atIndex:0];
    [computeEncoder setBuffer:impl_->bufferVector offset:0 atIndex:1];
    [computeEncoder setBuffer:impl_->bufferOutput offset:0 atIndex:2];
    [computeEncoder setBytes:&impl_->gemvShape.rows length:sizeof(uint) atIndex:3];
    [computeEncoder setBytes:&impl_->gemvShape.cols length:sizeof(uint) atIndex:4];

    MTLSize gridSize = MTLSizeMake(impl_->gemvShape.rows, 1, 1);
    
    NSUInteger upperBound = impl_->pipeline_state.maxTotalThreadsPerThreadgroup;
    if (upperBound > gridSize.width) {
        upperBound = gridSize.width;
    }
    MTLSize threadGroupSize = MTLSizeMake(upperBound, 1, 1);

    [computeEncoder dispatchThreads:gridSize threadsPerThreadgroup:threadGroupSize];
    [computeEncoder endEncoding];

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

void GemvKernel::download(std::span<float> output) {
    if (output.size() < impl_->gemvShape.rows) {
        throw std::runtime_error(
            "Invalid output size. Expected: "
            + std::to_string(impl_->gemvShape.rows)
            + ", got: " + std::to_string(output.size())
        );
    }
    std::memcpy(output.data(), [impl_->bufferOutput contents], impl_->gemvShape.rows * sizeof(float));
}

}