#include "llmetal/tensor.hpp"
#include "llmetal/embedding.hpp"
#include "llmetal/metal_context.hpp"
#include "llmetal/metal_job.hpp"

#include <Foundation/Foundation.h>
#include <Metal/Metal.h>

namespace llmetal {

class EmbeddingKernel::Impl {
public:
    explicit Impl(MetalContext& context): metalContext(context) {};
    MetalContext& metalContext;
private:
    id<MTLComputePipelineState> pipeline_state = nil;
    id<MTLCommandBuffer> lastCommandBuffer = nil;
friend class EmbeddingKernel;
};

EmbeddingKernel::EmbeddingKernel(MetalContext& context)
    : impl_(std::make_unique<Impl>(context)) {
    NSError* error = nil;
    id<MTLDevice> device = (__bridge id<MTLDevice>)context.device_handle();

    // We need to manually get the library from the shaders folder
    NSURL* exeURL = [[NSBundle mainBundle] executableURL];
    NSURL* libURL = [
        [exeURL URLByDeletingLastPathComponent] URLByAppendingPathComponent:@"shaders/embedding.metallib"
    ];  
    id<MTLLibrary> library = [device newLibraryWithURL:libURL error:&error];
    if (library == nil) {
        throw std::runtime_error(
            std::string("Library error: ") + [error.localizedDescription UTF8String]
        );
    }

    id<MTLFunction> function = [library newFunctionWithName:@"embedding_naive"];
    if (function == nil) throw std::runtime_error("Function not found");

    impl_->pipeline_state = [device newComputePipelineStateWithFunction:function error:&error];
    if (impl_->pipeline_state == nil) {
        throw std::runtime_error(
            std::string("Failed to create pipeline state: ") 
            + [error.localizedDescription UTF8String]
        );
    }
}
    
EmbeddingKernel::~EmbeddingKernel() = default;
EmbeddingKernel::EmbeddingKernel(EmbeddingKernel&&) noexcept = default;
EmbeddingKernel& EmbeddingKernel::operator=(EmbeddingKernel&&) noexcept = default;


MetalJob EmbeddingKernel::submit_repeated(
    std::size_t repeats,
    const GpuTensor<float>& table,       // [vocab_size, hidden]
    const GpuTensor<std::uint32_t>& ids, // [batch_size, sequence_length]
    const GpuTensor<float>& output       // [batch_size, sequence_length, hidden]
) {
    uint hidden_size = table.shape()[1];
    uint n_input_tokens = ids.shape()[0] * ids.shape()[1];
    if (n_input_tokens == 0) throw std::runtime_error("Invalid input size");
    if (hidden_size == 0) throw std::runtime_error("Invalid hidden size");
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
    [computeEncoder setBuffer:(__bridge id<MTLBuffer>)table.buffer_handle()  offset:table.byte_offset_  atIndex:0];
    [computeEncoder setBuffer:(__bridge id<MTLBuffer>)ids.buffer_handle()    offset:ids.byte_offset_    atIndex:1];
    [computeEncoder setBuffer:(__bridge id<MTLBuffer>)output.buffer_handle() offset:output.byte_offset_ atIndex:2];
    [computeEncoder setBytes:&hidden_size length:sizeof(uint) atIndex:3];


    // Naive - just one thread per row
    MTLSize gridSize = MTLSizeMake(n_input_tokens, 1, 1);
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

MetalJob EmbeddingKernel::submit(
    const GpuTensor<float>& table,       // [vocab_size, hidden]
    const GpuTensor<std::uint32_t>& ids, // [batch_size, sequence_length]
    const GpuTensor<float>& output       // [batch_size, sequence_length, hidden]
) {
    return submit_repeated(1, table, ids, output);
}

bool EmbeddingKernel::in_progress() const noexcept {
    if (impl_->lastCommandBuffer == nil) return false;
    const auto status = [impl_->lastCommandBuffer status];
    return status != MTLCommandBufferStatusCompleted && 
           status != MTLCommandBufferStatusError;
}


} // namespace llmetal
