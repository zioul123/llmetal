
#include <iostream>
#include <objc/NSObjCRuntime.h>
#include <stdexcept>
#include <cstddef>
#include <cstring>
#include <string>
#include <vector>
#include <algorithm>

#include <llmetal/metal_context.hpp>
#include <llmetal/gemv_interleaved.hpp>

#include <Foundation/Foundation.h>
#include <Metal/Metal.h>

namespace llmetal {

class GemvInterleavedKernel::Impl {
public:
    explicit Impl(MetalContext& context): metalContext(context) {}
    MetalContext& metalContext;
private:
    id<MTLComputePipelineState> pipeline_state = nil;
    id<MTLCommandBuffer> lastCommandBuffer = nil;
    id<MTLBuffer> bufferMatrix = nil;
    id<MTLBuffer> bufferVector = nil;
    id<MTLBuffer> bufferOutput = nil;
    std::size_t capacityMatrix = 0;
    std::size_t capacityVector = 0;
    std::size_t capacityOutput = 0;
    GemvShape gemvShape = {0, 0};
friend class GemvInterleavedKernel;
};

GemvInterleavedKernel::GemvInterleavedKernel(MetalContext& context)
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
    id<MTLFunction> function = [library newFunctionWithName:@"gemv_interleaved"];
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

GemvInterleavedKernel::~GemvInterleavedKernel() = default;
GemvInterleavedKernel::GemvInterleavedKernel(GemvInterleavedKernel&&) noexcept = default;
GemvInterleavedKernel& GemvInterleavedKernel::operator=(GemvInterleavedKernel&&) noexcept = default;

void GemvInterleavedKernel::prepare(GemvShape shape) {
    if (in_progress()) throw std::runtime_error("Kernel is already in progress");
    if (shape.cols == 0 || shape.rows == 0) throw std::runtime_error("Invalid gemv shape");
    
    std::size_t matElements = shape.cols * shape.rows;
    id<MTLDevice> device = (__bridge id<MTLDevice>)impl_->metalContext.device_handle();

    if (matElements > impl_->capacityMatrix) {
        id<MTLBuffer> bufferMatrix = [device newBufferWithLength: matElements * sizeof(float) options: MTLResourceStorageModeShared];
        if (bufferMatrix == nil ) {
            throw std::runtime_error("Failed to create matrix buffer");
        }
        impl_->bufferMatrix = bufferMatrix;
        impl_->capacityMatrix = matElements;
    }
    if (shape.cols > impl_->capacityVector) {
        id<MTLBuffer> bufferVector = [device newBufferWithLength: shape.cols * sizeof(float) options: MTLResourceStorageModeShared];
        if (bufferVector == nil ) {
            throw std::runtime_error("Failed to create vector buffer");
        }
        impl_->bufferVector = bufferVector;
        impl_->capacityVector = shape.cols;
    }

    if (shape.rows > impl_->capacityOutput) {
        id<MTLBuffer> bufferOutput = [device newBufferWithLength: shape.rows * sizeof(float) options: MTLResourceStorageModeShared];
        if (bufferOutput == nil) {
            throw std::runtime_error("Failed to create buffers");
        }
        impl_->bufferOutput = bufferOutput;
        impl_->capacityOutput = shape.rows;
    }

    impl_->gemvShape = shape;
}

void GemvInterleavedKernel::upload_matrix(std::span<const float> matrix) {
    if (in_progress()) throw std::runtime_error("Kernel is already in progress");
    if (matrix.size() != impl_->gemvShape.cols * impl_->gemvShape.rows) {
        throw std::runtime_error(
            "Invalid matrix size. Expected: "
            + std::to_string(impl_->gemvShape.cols * impl_->gemvShape.rows)
            + ", got: " + std::to_string(matrix.size())
        );
    }

    std::memcpy([impl_->bufferMatrix contents], matrix.data(), matrix.size() * sizeof(float));
}

void GemvInterleavedKernel::upload_vector(std::span<const float> vector) {
    if (in_progress()) throw std::runtime_error("Kernel is already in progress");
    if (vector.size() != impl_->gemvShape.cols) {
        throw std::runtime_error(
            "Invalid vector size. Expected: "
            + std::to_string(impl_->gemvShape.cols)
            + ", got: " + std::to_string(vector.size())
        );
    }
    std::memcpy([impl_->bufferVector contents], vector.data(), vector.size() * sizeof(float));
}

llmetal::MetalJob GemvInterleavedKernel::submit_repeated(std::size_t repeats) {
    if (repeats == 0) throw std::runtime_error("Invalid repeats");
    if (in_progress()) throw std::runtime_error("Kernel is already in progress");
    if (impl_->gemvShape.cols == 0 || impl_->gemvShape.rows == 0) throw std::runtime_error("Invalid gemv shape");

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

    MTLSize threadGroups = MTLSizeMake(impl_->gemvShape.rows, 1, 1);
    MTLSize threadGroupSize = MTLSizeMake([impl_->pipeline_state threadExecutionWidth], 1, 1);

    for (std::size_t i = 0; i < repeats; ++i) {
        [computeEncoder dispatchThreadgroups:threadGroups 
                       threadsPerThreadgroup:threadGroupSize];
    }

    [computeEncoder endEncoding];

    // Execute command
    [commandBuffer commit];
    impl_->lastCommandBuffer = commandBuffer;
    return llmetal::MetalJob((__bridge void*) commandBuffer);
}

llmetal::MetalJob GemvInterleavedKernel::submit() {
    return submit_repeated(1);
}

void GemvInterleavedKernel::download(std::span<float> output) {
    if (in_progress()) throw std::runtime_error("Kernel is already in progress");
    if (output.size() < impl_->gemvShape.rows) {
        throw std::runtime_error(
            "Invalid output size. Expected: "
            + std::to_string(impl_->gemvShape.rows)
            + ", got: " + std::to_string(output.size())
        );
    }
    std::memcpy(output.data(), [impl_->bufferOutput contents], impl_->gemvShape.rows * sizeof(float));
}

bool GemvInterleavedKernel::in_progress() const noexcept {
    if (impl_->lastCommandBuffer == nil) {
        return false;
    }
    const auto status = impl_->lastCommandBuffer.status;
    return status != MTLCommandBufferStatusCompleted &&
           status != MTLCommandBufferStatusError;
}

}
