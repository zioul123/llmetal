#include "llmetal/tensor.hpp"
#include "llmetal/rms_norm.hpp"
#include "llmetal/metal_context.hpp"
#include "llmetal/metal_job.hpp"

#include <Foundation/Foundation.h>
#include <Metal/Metal.h>

namespace llmetal {

class RmsNormKernel::Impl {
public:
    explicit Impl(MetalContext& context, std::uint32_t tptg, std::uint32_t tpr)
        : metalContext(context), tptg(tptg), tpr(tpr) {};
    MetalContext& metalContext;
    std::uint32_t tptg; // threads per threadgroup
    std::uint32_t tpr;  // threads per row
private:
    id<MTLComputePipelineState> pipeline_state = nil;
    id<MTLCommandBuffer> lastCommandBuffer = nil;
friend class RmsNormKernel;
};

RmsNormKernel::RmsNormKernel(MetalContext& context, std::uint32_t tptg, std::uint32_t tpr)
    : impl_(std::make_unique<Impl>(context, tptg, tpr)) {
    NSError* error = nil;
    id<MTLDevice> device = (__bridge id<MTLDevice>)context.device_handle();

    // We need to manually get the library from the shaders folder
    NSURL* exeURL = [[NSBundle mainBundle] executableURL];
    NSURL* libURL = [
        [exeURL URLByDeletingLastPathComponent] URLByAppendingPathComponent:@"shaders/rms_norm.metallib"
    ];  
    id<MTLLibrary> library = [device newLibraryWithURL:libURL error:&error];
    if (library == nil) {
        throw std::runtime_error(
            std::string("Library error: ") + [error.localizedDescription UTF8String]
        );
    }

    if (tptg % tpr != 0) {
        throw std::runtime_error("tptg must be a multiple of tpr");
    }

    id<MTLFunction> function = [library newFunctionWithName:@"rms_norm_naive"];
    if (function == nil) throw std::runtime_error("Function not found");

    impl_->pipeline_state = [device newComputePipelineStateWithFunction:function error:&error];
    if (impl_->pipeline_state == nil) {
        throw std::runtime_error(std::string("Failed to create pipeline state: ") 
                                 + [error.localizedDescription UTF8String]);
    }
}
    
RmsNormKernel::~RmsNormKernel() = default;
RmsNormKernel::RmsNormKernel(RmsNormKernel&&) noexcept = default;
RmsNormKernel& RmsNormKernel::operator=(RmsNormKernel&&) noexcept = default;


MetalJob RmsNormKernel::submit_repeated(
    std::size_t repeats,
    const GpuTensor<float>& input,   // [batch_size, sequence_length, hidden]
    const GpuTensor<float>& weights, // [hidden]
    GpuTensor<float>& output,        // [batch_size, sequence_length, hidden]
    const float epsilon
) {
    // Pull out shape information
    std::uint32_t hidden_size = checked_u32(weights.shape()[0], "rms_norm_hidden_size");
    std::uint32_t batch_size = checked_u32(input.shape()[0], "rms_norm_batch_size");
    std::uint32_t sequence_length = checked_u32(input.shape()[1], "rms_norm_sequence_length");
    std::uint32_t n_input = checked_multiply(batch_size, sequence_length);
    if (n_input == 0) throw std::runtime_error("Invalid input size 0");
    if (hidden_size == 0) throw std::runtime_error("Invalid hidden size");

    if (input.shape().rank() != 3) throw std::runtime_error("Invalid input shape, expected [batch_size, sequence_length, hidden_size]");
    if (n_input * hidden_size != input.numel()) throw std::runtime_error("Invalid input size. Expected" + std::to_string(n_input * hidden_size) + " but got " + std::to_string(input.numel()));
    if (weights.shape().rank() != 1) throw std::runtime_error("Invalid weights shape, expected [hidden_size]");
    if (output.shape().rank() != 3) throw std::runtime_error("Invalid output shape, expected [batch_size, sequence_length, hidden_size]");
    if (output.shape()[0] != batch_size || output.shape()[1] != sequence_length || output.shape()[2] != hidden_size) {
        throw std::runtime_error("Invalid output size");
    }
    if (repeats == 0) throw std::runtime_error("Invalid repeats");
    if (in_progress()) throw std::runtime_error("Kernel is already in progress");

    id<MTLCommandQueue> commandQueue = (__bridge id<MTLCommandQueue>)impl_->metalContext.command_queue_handle();

    // Create command buffer and encoder
    id<MTLCommandBuffer> commandBuffer = [commandQueue commandBuffer];
    if (commandBuffer == nil) throw std::runtime_error("Could not create command buffer");
    id<MTLComputeCommandEncoder> computeEncoder = [commandBuffer computeCommandEncoder];
    if (computeEncoder == nil) throw std::runtime_error("Could not create compute command encoder");

    // Encode command
    [computeEncoder setComputePipelineState:impl_->pipeline_state];
    [computeEncoder setBuffer:(__bridge id<MTLBuffer>)input.buffer_handle()   offset:input.byte_offset_   atIndex:0];
    [computeEncoder setBuffer:(__bridge id<MTLBuffer>)weights.buffer_handle() offset:weights.byte_offset_ atIndex:1];
    [computeEncoder setBuffer:(__bridge id<MTLBuffer>)output.buffer_handle()  offset:output.byte_offset_  atIndex:2];
    [computeEncoder setBytes:&hidden_size length:sizeof(uint) atIndex:3];
    [computeEncoder setBytes:&epsilon length:sizeof(float) atIndex:4];

    if (impl_->tpr == 1) {
        // Naive - just one thread per row
        MTLSize gridSize = MTLSizeMake(n_input, 1, 1);
        NSUInteger upperBound = impl_->tptg;
        upperBound = (upperBound > gridSize.width) ? gridSize.width : upperBound;
        MTLSize threadGroupSize = MTLSizeMake(upperBound, 1, 1);

        for (std::size_t i = 0; i < repeats; ++i) {
            [computeEncoder dispatchThreads:gridSize threadsPerThreadgroup:threadGroupSize];
        }

        [computeEncoder endEncoding];
        [commandBuffer commit];
        impl_->lastCommandBuffer = commandBuffer;
        return llmetal::MetalJob((__bridge void*) commandBuffer);
    } else if (impl_->tpr < impl_->pipeline_state.threadExecutionWidth) {
        throw std::runtime_error("Not implemented");
    } else {
        throw std::runtime_error("Not implemented");
    }
}

MetalJob RmsNormKernel::submit(
    const GpuTensor<float>& input,   // [batch_size, sequence_length, hidden]
    const GpuTensor<float>& weights, // [hidden]
    GpuTensor<float>& output,        // [batch_size, sequence_length, hidden]
    const float epsilon
) {
    return submit_repeated(1, input, weights, output, epsilon);
}

bool RmsNormKernel::in_progress() const noexcept {
    if (impl_->lastCommandBuffer == nil) return false;
    const auto status = [impl_->lastCommandBuffer status];
    return status != MTLCommandBufferStatusCompleted && 
           status != MTLCommandBufferStatusError;
}


} // namespace llmetal
