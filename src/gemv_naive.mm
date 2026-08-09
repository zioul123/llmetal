#include "llmetal/tensor.hpp"
#include <iostream>
#include <stdexcept>
#include <cstddef>
#include <cstring>
#include <string>
#include <algorithm>

#include <llmetal/metal_context.hpp>
#include <llmetal/gemv_naive.hpp>

#include <Foundation/Foundation.h>
#include <Metal/Metal.h>

namespace llmetal {

class GemvNaiveKernel::Impl {
public:
    explicit Impl(MetalContext& context): metalContext(context) {}
    MetalContext& metalContext;
private:
    id<MTLComputePipelineState> pipeline_state = nil;
    id<MTLCommandBuffer> lastCommandBuffer = nil;
friend class GemvNaiveKernel;
};

GemvNaiveKernel::GemvNaiveKernel(MetalContext& context)
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
    id<MTLFunction> function = [library newFunctionWithName:@"gemv_naive"];
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

GemvNaiveKernel::~GemvNaiveKernel() = default;
GemvNaiveKernel::GemvNaiveKernel(GemvNaiveKernel&&) noexcept = default;
GemvNaiveKernel& GemvNaiveKernel::operator=(GemvNaiveKernel&&) noexcept = default;


llmetal::MetalJob GemvNaiveKernel::submit_repeated(
    std::size_t repeats,
    const GpuTensor<float>& matrix, // [rows, cols]
    const GpuTensor<float>& vector, // [cols]
    GpuTensor<float>& output        // [rows]
) {
    std::size_t rows = checked_u32(matrix.shape()[0], "matrix_rows");
    std::size_t cols = checked_u32(matrix.shape()[1], "matrix_cols");
    if (repeats == 0) throw std::runtime_error("Invalid repeats");
    if (in_progress()) throw std::runtime_error("Kernel is already in progress");
    if (rows == 0 || cols == 0) throw std::runtime_error("Invalid gemv shape");
    if (rows != output.shape()[0]) {
        throw std::runtime_error(
            "Invalid output size. Expected: " + std::to_string(rows) + ", got: " + std::to_string(output.shape()[0])
        );
    }
    if (cols != vector.shape()[0]) {
        throw std::runtime_error(
            "Invalid vector size. Expected: " + std::to_string(cols) + ", got: " + std::to_string(vector.shape()[0])
        );
    }

    id<MTLCommandQueue> commandQueue = (__bridge id<MTLCommandQueue>)impl_->metalContext.command_queue_handle();

    // Create command buffer and encoder
    id<MTLCommandBuffer> commandBuffer = [commandQueue commandBuffer];
    if (commandBuffer == nil) throw std::runtime_error("Could not create command buffer");
    id<MTLComputeCommandEncoder> computeEncoder = [commandBuffer computeCommandEncoder];
    if (computeEncoder == nil) throw std::runtime_error("Could not create compute command encoder");

    // Encode command
    [computeEncoder setComputePipelineState:impl_->pipeline_state];
    [computeEncoder setBuffer:(__bridge id<MTLBuffer>)matrix.buffer_handle() offset:0 atIndex:0];
    [computeEncoder setBuffer:(__bridge id<MTLBuffer>)vector.buffer_handle() offset:0 atIndex:1];
    [computeEncoder setBuffer:(__bridge id<MTLBuffer>)output.buffer_handle() offset:0 atIndex:2];
    [computeEncoder setBytes:&rows length:sizeof(uint) atIndex:3];
    [computeEncoder setBytes:&cols length:sizeof(uint) atIndex:4];

    MTLSize gridSize = MTLSizeMake(rows, 1, 1);
    // NSUInteger upperBound = impl_->pipeline_state.maxTotalThreadsPerThreadgroup;
    NSUInteger upperBound = [impl_->pipeline_state threadExecutionWidth];
    upperBound = (upperBound > gridSize.width) ? gridSize.width : upperBound;
    MTLSize threadGroupSize = MTLSizeMake(upperBound, 1, 1);

    for (std::size_t i = 0; i < repeats; ++i) {
        [computeEncoder dispatchThreads:gridSize threadsPerThreadgroup:threadGroupSize];
    }

    [computeEncoder endEncoding];

    // Execute command
    [commandBuffer commit];
    impl_->lastCommandBuffer = commandBuffer;
    return llmetal::MetalJob((__bridge void*) commandBuffer);
}

llmetal::MetalJob GemvNaiveKernel::submit(
    const GpuTensor<float>& matrix, // [rows, cols]
    const GpuTensor<float>& vector, // [cols]
    GpuTensor<float>& output        // [rows]
) {
    return submit_repeated(1, matrix, vector, output);
}

bool GemvNaiveKernel::in_progress() const noexcept {
    if (impl_->lastCommandBuffer == nil) {
        return false;
    }
    const auto status = impl_->lastCommandBuffer.status;
    return status != MTLCommandBufferStatusCompleted &&
           status != MTLCommandBufferStatusError;
}

}
