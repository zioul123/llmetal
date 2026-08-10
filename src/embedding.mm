#include "llmetal/tensor.hpp"
#include "llmetal/embedding.hpp"
#include "llmetal/metal_context.hpp"
#include "llmetal/metal_job.hpp"

#include <Foundation/Foundation.h>
#include <Metal/Metal.h>

namespace llmetal {

class EmbeddingKernel::Impl {
public:
    explicit Impl(MetalContext& context, std::uint32_t tptg, std::uint32_t tpr)
        : metalContext(context), tptg(tptg), tpr(tpr) {};
    MetalContext& metalContext;
    std::uint32_t tptg; // threads per threadgroup
    std::uint32_t tpr;  // threads per row
private:
    id<MTLComputePipelineState> pipeline_state = nil;
    id<MTLCommandBuffer> lastCommandBuffer = nil;
friend class EmbeddingKernel;
};

EmbeddingKernel::EmbeddingKernel(MetalContext& context, std::uint32_t tptg, std::uint32_t tpr)
    : impl_(std::make_unique<Impl>(context, tptg, tpr)) {
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

    if (tptg % tpr != 0) {
        throw std::runtime_error("tptg must be a multiple of tpr");
    }

    id<MTLFunction> function = [library newFunctionWithName:@"embedding_naive"];
    if (function == nil) throw std::runtime_error("Function not found");

    impl_->pipeline_state = [device newComputePipelineStateWithFunction:function error:&error];
    if (impl_->pipeline_state == nil) {
        throw std::runtime_error(std::string("Failed to create pipeline state: ") 
                                 + [error.localizedDescription UTF8String]);
    }

    // Very dumb - just wanted the pipeline state to be created to get execution width
    if (tpr != 1) {
        // Validate that tpr, if more than the size of a simd group, is a multiple. 
        // TODO: Unfortunately, this must be done after we created the pipeline state, so it's in a weird place.
        if (tpr >= impl_->pipeline_state.threadExecutionWidth && 
            tpr % impl_->pipeline_state.threadExecutionWidth != 0
        ) {
            throw std::runtime_error("tpr must be less than or a multiple of the thread execution width (SIMD Group size)");
        }
        function = tpr < impl_->pipeline_state.threadExecutionWidth 
                 ? [library newFunctionWithName:@"embedding_tpr"]
                 : [library newFunctionWithName:@"embedding_nsgpr"];

        if (function == nil) throw std::runtime_error("Function not found");

        impl_->pipeline_state = [device newComputePipelineStateWithFunction:function error:&error];
        if (impl_->pipeline_state == nil) {
            throw std::runtime_error(
                std::string("Failed to create pipeline state: ") 
                + [error.localizedDescription UTF8String]
            );
        }
    }

}
    
EmbeddingKernel::~EmbeddingKernel() = default;
EmbeddingKernel::EmbeddingKernel(EmbeddingKernel&&) noexcept = default;
EmbeddingKernel& EmbeddingKernel::operator=(EmbeddingKernel&&) noexcept = default;


MetalJob EmbeddingKernel::submit_repeated(
    std::size_t repeats,
    const GpuTensor<float>& table,       // [vocab_size, hidden]
    const GpuTensor<std::uint32_t>& ids, // [batch_size, sequence_length]
    GpuTensor<float>& output            // [batch_size, sequence_length, hidden]
) {
    // Pull out shape information
    std::uint32_t vocab_size = checked_u32(table.shape()[0], "vocab_size");
    std::size_t hidden_size = table.shape()[1];
    std::size_t batch_size = ids.shape()[0];
    std::size_t sequence_length = ids.shape()[1];
    std::size_t n_input_tokens = checked_multiply(batch_size, sequence_length);

    if (n_input_tokens == 0) throw std::runtime_error("Invalid input size");
    if (hidden_size == 0) throw std::runtime_error("Invalid hidden size");
    
    if (vocab_size * hidden_size != table.numel()) throw std::runtime_error("Invalid table size");
    if (table.shape().rank() != 2) throw std::runtime_error("Invalid table shape, expected [vocab_size, hidden]");
    
    if (n_input_tokens != ids.numel()) throw std::runtime_error("Invalid ids size");
    if (ids.shape().rank() != 2) throw std::runtime_error("Invalid ids shape, expected [batch_size, sequence_length]");
    
    if (output.shape()[0] != batch_size || output.shape()[1] != sequence_length || output.shape()[2] != hidden_size) {
        throw std::runtime_error("Invalid output size");
    }
    if (output.shape().rank() != 3) throw std::runtime_error("Invalid output shape, expected [batch_size, sequence_length, hidden]");

    if (repeats == 0) throw std::runtime_error("Invalid repeats");
    if (in_progress()) throw std::runtime_error("Kernel is already in progress");

    const std::uint32_t hidden_size_u32 = checked_u32(hidden_size, "hidden_size");
    const std::uint32_t n_input_tokens_u32 = checked_u32(n_input_tokens, "n_input_tokens");

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
    [computeEncoder setBytes:&hidden_size_u32 length:sizeof(uint) atIndex:3];
    [computeEncoder setBytes:&vocab_size      length:sizeof(uint) atIndex:4];

    if (impl_->tpr == 1) {
        // Naive - just one thread per row
        MTLSize gridSize = MTLSizeMake(n_input_tokens, 1, 1);
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
        // some threads per row
        throw std::runtime_error("Not implemented");
    } else {
        // Rows per threadgroup
        NSUInteger rptg = impl_->tptg / impl_->tpr;
        // SIMD groups per row
        NSUInteger sgpr = impl_->tpr / impl_->pipeline_state.threadExecutionWidth;
        [computeEncoder setBytes:&rptg               length:sizeof(uint) atIndex:5];
        [computeEncoder setBytes:&impl_->tpr         length:sizeof(uint) atIndex:6];
        [computeEncoder setBytes:&n_input_tokens_u32 length:sizeof(uint) atIndex:7];
        [computeEncoder setBytes:&sgpr               length:sizeof(uint) atIndex:8];

        // Number of thread groups is ceil_div(rows, rows per threadgroup)
        MTLSize gridSize = MTLSizeMake(1 , (n_input_tokens + rptg - 1) / rptg, 1);
        MTLSize threadGroupSize = MTLSizeMake(impl_->tpr, rptg, 1);

        for (std::size_t i = 0; i < repeats; ++i) {
            [computeEncoder dispatchThreadgroups:gridSize threadsPerThreadgroup:threadGroupSize];
        }

        [computeEncoder endEncoding];
        [commandBuffer commit];
        impl_->lastCommandBuffer = commandBuffer;
        return llmetal::MetalJob((__bridge void*) commandBuffer);
    }
}

MetalJob EmbeddingKernel::submit(
    const GpuTensor<float>& table,       // [vocab_size, hidden]
    const GpuTensor<std::uint32_t>& ids, // [batch_size, sequence_length]
    GpuTensor<float>& output      // [batch_size, sequence_length, hidden]
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
