#include <iostream>
#include <objc/NSObjCRuntime.h>
#include <stdexcept>
#include <cstddef>
#include <cstring>
#include <string>
#include <vector>
#include <algorithm>

#include <llmetal/metal_context.hpp>
#include <llmetal/gemv_NRPSG.hpp>

#include <Foundation/Foundation.h>
#include <Metal/Metal.h>

namespace llmetal {

class GemvNRPSGKernel::Impl {
public:
    explicit Impl(MetalContext& context, std::size_t rpsg, std::size_t sgptg): 
        metalContext(context), rpsg(rpsg), sgptg(sgptg) {}
    MetalContext& metalContext;
    std::size_t rpsg; // Rows per SIMD group
    std::size_t sgptg; // SIMD groups per threadgroup
private:
    id<MTLComputePipelineState> pipeline_state = nil;
    id<MTLCommandBuffer> lastCommandBuffer = nil;
friend class GemvNRPSGKernel;
};

GemvNRPSGKernel::GemvNRPSGKernel(MetalContext& context, std::size_t rpsg, std::size_t sgptg)
    : impl_(std::make_unique<Impl>(context, rpsg, sgptg)) {
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

    // From `shaders/gemv.metal` - choose the shader based on desired rpsg
    id<MTLFunction> function = 
        rpsg == 1 ? [library newFunctionWithName:@"gemv_1RPSG"]
        : rpsg == 2 ? [library newFunctionWithName:@"gemv_2RPSG"]
        : rpsg == 4 ? [library newFunctionWithName:@"gemv_4RPSG"]
        : rpsg == 8 ? [library newFunctionWithName:@"gemv_8RPSG"]
        : throw std::runtime_error("Invalid rpsg, expected 1, 2, 4, or 8");
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

GemvNRPSGKernel::~GemvNRPSGKernel() = default;
GemvNRPSGKernel::GemvNRPSGKernel(GemvNRPSGKernel&&) noexcept = default;
GemvNRPSGKernel& GemvNRPSGKernel::operator=(GemvNRPSGKernel&&) noexcept = default;

llmetal::MetalJob GemvNRPSGKernel::submit_repeated(
    std::size_t repeats,
    const GpuTensor<float>& matrix, // [rows, cols]
    const GpuTensor<float>& vector, // [cols]
    GpuTensor<float>& output        // [rows]
) {
    std::size_t rows = checked_u32(matrix.shape()[0], "matrix_rows");
    std::size_t cols = checked_u32(matrix.shape()[1], "matrix_cols");

    if (repeats == 0) throw std::runtime_error("Invalid repeats");
    if (in_progress()) throw std::runtime_error("Kernel is already in progress");
    if (cols == 0 || rows == 0) throw std::runtime_error("Invalid gemv shape");
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

    // sgptg simd groups per threadgroup, and rpsg rows per simd group
    NSUInteger rows_per_threadgroup = impl_->sgptg * impl_->rpsg;
    // so we have ceil_div(rows, rows_per_threadgroup) threadgroups
    MTLSize threadGroups = MTLSizeMake((rows + rows_per_threadgroup - 1) / rows_per_threadgroup, 1, 1);
    MTLSize threadGroupSize = MTLSizeMake([impl_->pipeline_state threadExecutionWidth] * impl_->sgptg, 1, 1);

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

llmetal::MetalJob GemvNRPSGKernel::submit(
    const GpuTensor<float>& matrix, // [rows, cols]
    const GpuTensor<float>& vector, // [cols]
    GpuTensor<float>& output        // [rows]
) {
    return submit_repeated(1, matrix, vector, output);
}

bool GemvNRPSGKernel::in_progress() const noexcept {
    if (impl_->lastCommandBuffer == nil) return false;
    const auto status = impl_->lastCommandBuffer.status;
    return status != MTLCommandBufferStatusCompleted &&
           status != MTLCommandBufferStatusError;
}

}
