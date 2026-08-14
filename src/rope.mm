#include "llmetal/tensor.hpp"
#include "llmetal/rope.hpp"
#include "llmetal/metal_context.hpp"
#include "llmetal/metal_job.hpp"

#include <Foundation/Foundation.h>
#include <Metal/Metal.h>

namespace llmetal {

class RoPEKernel::Impl {
public:
    explicit Impl(MetalContext& context, std::uint32_t tptg, std::uint32_t tpr)
        : metalContext(context), tptg(tptg), tpr(tpr) {};
    MetalContext& metalContext;
    std::uint32_t tptg; // threads per threadgroup
    std::uint32_t tpr;  // threads per row
private:
    id<MTLComputePipelineState> pipeline_state = nil;
    id<MTLCommandBuffer> lastCommandBuffer = nil;
friend class RoPEKernel;
};

RoPEKernel::RoPEKernel(MetalContext& context, std::uint32_t tptg, std::uint32_t tpr)
    : impl_(std::make_unique<Impl>(context, tptg, tpr)) {
    NSError* error = nil;
    id<MTLDevice> device = (__bridge id<MTLDevice>)context.device_handle();

    // We need to manually get the library from the shaders folder
    NSURL* exeURL = [[NSBundle mainBundle] executableURL];
    NSURL* libURL = [
        [exeURL URLByDeletingLastPathComponent] URLByAppendingPathComponent:@"shaders/rope.metallib"
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

    id<MTLFunction> function = [library newFunctionWithName:@"rope_naive"];
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
        function = tpr == impl_->pipeline_state.threadExecutionWidth 
                 ? [library newFunctionWithName:@"rope_1sgpr"]
                 : [library newFunctionWithName:@"rope_nsgpr"];

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
    
RoPEKernel::~RoPEKernel() = default;
RoPEKernel::RoPEKernel(RoPEKernel&&) noexcept = default;
RoPEKernel& RoPEKernel::operator=(RoPEKernel&&) noexcept = default;


MetalJob RoPEKernel::submit_repeated(
    std::size_t repeats,
    const GpuTensor<float>& input,      // [batch, seq_length, num_heads, head_dim]
    GpuTensor<float>& output,           // [batch, seq_length, num_heads, head_dim]
    const GpuTensor<float>& cos_and_sin // [max_seq_length, rotary_dim / 2, 2 (cos_and_sin)]
) {
    if (repeats == 0) throw std::runtime_error("Invalid repeats");
    if (in_progress()) throw std::runtime_error("Kernel is already in progress");

    if (input.shape().rank() != 4 || 
        output.shape().rank() != 4 || 
        input.shape()[0] != output.shape()[0] || 
        input.shape()[1] != output.shape()[1] || 
        input.shape()[2] != output.shape()[2] ||
        input.shape()[3] != output.shape()[3]
    ) {
        throw std::invalid_argument("input and output must be 4D tensors of the same shape");
    }

    if (cos_and_sin.shape().rank() != 3 || cos_and_sin.shape()[2] != 2) {
        throw std::invalid_argument("cos_and_sin must be a 3D tensor of shape [max_seq_length, rotary_dim / 2, 2]");
    }

    // Pull out shape information
    std::uint32_t B = checked_u32(input.shape()[0], "rms_norm_batch_size");
    std::uint32_t S = checked_u32(input.shape()[1], "rms_norm_sequence_length");
    std::uint32_t H = checked_u32(input.shape()[2], "num_heads");
    std::uint32_t D = checked_u32(input.shape()[3], "head_dim");
    std::uint32_t P = checked_u32(cos_and_sin.shape()[1], "rotary_pairs");
    std::uint32_t R = checked_multiply_u32(P, 2);
    
    std::uint32_t n_input = checked_multiply_u32(checked_multiply_u32(B, S), H);

    if (R == 0) throw std::invalid_argument("rotary_dim must be positive");
    if (R > D) throw std::invalid_argument("rotary_dim must be <= head_dim");
    if (cos_and_sin.shape()[0] < S) throw std::invalid_argument("cos_and_sin max_seq_length must be longer than sequence length");

    id<MTLCommandQueue> commandQueue = (__bridge id<MTLCommandQueue>)impl_->metalContext.command_queue_handle();

    // Create command buffer and encoder
    id<MTLCommandBuffer> commandBuffer = [commandQueue commandBuffer];
    if (commandBuffer == nil) throw std::runtime_error("Could not create command buffer");
    id<MTLComputeCommandEncoder> computeEncoder = [commandBuffer computeCommandEncoder];
    if (computeEncoder == nil) throw std::runtime_error("Could not create compute command encoder");

    // Encode command
    [computeEncoder setComputePipelineState:impl_->pipeline_state];
    [computeEncoder setBuffer:(__bridge id<MTLBuffer>)input.buffer_handle()       offset:input.byte_offset_       atIndex:0];
    [computeEncoder setBuffer:(__bridge id<MTLBuffer>)cos_and_sin.buffer_handle() offset:cos_and_sin.byte_offset_ atIndex:1];
    [computeEncoder setBuffer:(__bridge id<MTLBuffer>)output.buffer_handle()      offset:output.byte_offset_      atIndex:2];
    [computeEncoder setBytes:&S length:sizeof(uint) atIndex:3];
    [computeEncoder setBytes:&H length:sizeof(uint) atIndex:4];
    [computeEncoder setBytes:&D length:sizeof(uint) atIndex:5];
    [computeEncoder setBytes:&P length:sizeof(uint) atIndex:6];
    [computeEncoder setBytes:&R length:sizeof(uint) atIndex:7];

    if (impl_->tpr == 1) {
        // Naive - just one thread per row
        MTLSize gridSize = MTLSizeMake(n_input, 1, 1);
        NSUInteger upperBound = impl_->tptg;
        upperBound = (upperBound > gridSize.width) ? gridSize.width : upperBound;
        MTLSize threadGroupSize = MTLSizeMake(upperBound, 1, 1);
        for (std::size_t i = 0; i < repeats; ++i) {
            [computeEncoder dispatchThreads:gridSize threadsPerThreadgroup:threadGroupSize];
        }
    } 
    else if (impl_->tpr == impl_->pipeline_state.threadExecutionWidth) {
        // 1 SIMD group per row, rptg rows per threadgroup
        NSUInteger rptg = impl_->tptg / impl_->tpr;
        MTLSize gridSize = MTLSizeMake(impl_->tpr, n_input, 1);
        MTLSize threadGroupSize = MTLSizeMake(impl_->tpr, rptg, 1);
        for (std::size_t i = 0; i < repeats; ++i) {
            [computeEncoder dispatchThreads:gridSize threadsPerThreadgroup:threadGroupSize];
        }
    } 
    else if (impl_->tpr > impl_->pipeline_state.threadExecutionWidth) {
        // N SIMD group per row, 1 row per threadgroup
        NSUInteger sgptg = impl_->tpr / impl_->pipeline_state.threadExecutionWidth;
        MTLSize gridSize = MTLSizeMake(impl_->tpr, n_input, 1);
        MTLSize threadGroupSize = MTLSizeMake(impl_->tpr, 1, 1);
        [computeEncoder setBytes:&sgptg length:sizeof(uint) atIndex:8];
        for (std::size_t i = 0; i < repeats; ++i) {
            [computeEncoder dispatchThreads:gridSize threadsPerThreadgroup:threadGroupSize];
        }
    }
    else {
        // We don't support <32 per row, nor multiple simd groups and rows per threadgroup,
        // only allow eitehr multi simd per row or multi row per threadgroup with 1 simd group per row.
        throw std::runtime_error("Not implemented - tpr should be a multiple of the thread execution width");
    }
    [computeEncoder endEncoding];
    [commandBuffer commit];
    impl_->lastCommandBuffer = commandBuffer;
    return llmetal::MetalJob((__bridge void*) commandBuffer);
}

MetalJob RoPEKernel::submit(
    const GpuTensor<float>& input,      // [batch, seq_length, num_heads, head_dim]
    GpuTensor<float>& output,           // [batch, seq_length, num_heads, head_dim]
    const GpuTensor<float>& cos_and_sin // [max_seq_length, rotary_dim / 2, 2 (cos_and_sin)]
) {
    return submit_repeated(1, input, output, cos_and_sin);
}

bool RoPEKernel::in_progress() const noexcept {
    if (impl_->lastCommandBuffer == nil) return false;
    const auto status = [impl_->lastCommandBuffer status];
    return status != MTLCommandBufferStatusCompleted && 
           status != MTLCommandBufferStatusError;
}


} // namespace llmetal
