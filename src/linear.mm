#include "llmetal/tensor.hpp"
#include "llmetal/linear.hpp"
#include "llmetal/metal_context.hpp"
#include "llmetal/metal_job.hpp"

#include <Foundation/Foundation.h>
#include <Metal/Metal.h>

namespace llmetal {

class LinearKernel::Impl {
public:
    explicit Impl(MetalContext& context, std::uint32_t tptg, std::uint32_t tpr)
        : metalContext(context), tptg(tptg), tpr(tpr) {};
    MetalContext& metalContext;
    std::uint32_t tptg; // threads per threadgroup
    std::uint32_t tpr;  // threads per row
private:
    id<MTLComputePipelineState> pipeline_state_with_bias = nil;
    id<MTLComputePipelineState> pipeline_state_without_bias = nil;
    id<MTLCommandBuffer> lastCommandBuffer = nil;
friend class LinearKernel;
};

LinearKernel::LinearKernel(MetalContext& context, std::uint32_t tptg, std::uint32_t tpr)
    : impl_(std::make_unique<Impl>(context, tptg, tpr)) {
    NSError* error = nil;
    id<MTLDevice> device = (__bridge id<MTLDevice>)context.device_handle();

    // We need to manually get the library from the shaders folder
    NSURL* exeURL = [[NSBundle mainBundle] executableURL];
    NSURL* libURL = [
        [exeURL URLByDeletingLastPathComponent] URLByAppendingPathComponent:@"shaders/linear.metallib"
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

    id<MTLFunction> function_with_bias = [library newFunctionWithName:@"linear_naive_with_bias"];
    if (function_with_bias == nil) throw std::runtime_error("Function linear_naive_with_bias not found");
    id<MTLFunction> function_without_bias = [library newFunctionWithName:@"linear_naive_without_bias"];
    if (function_with_bias == nil) throw std::runtime_error("Function linear_naive_without_bias not found");

    impl_->pipeline_state_with_bias = [device newComputePipelineStateWithFunction:function_with_bias error:&error];
    if (impl_->pipeline_state_with_bias == nil) {
        throw std::runtime_error(std::string("Failed to create pipeline state: ") 
                                 + [error.localizedDescription UTF8String]);
    }

    impl_->pipeline_state_without_bias = [device newComputePipelineStateWithFunction:function_without_bias error:&error];
    if (impl_->pipeline_state_without_bias == nil) {
        throw std::runtime_error(std::string("Failed to create pipeline state: ") 
                                 + [error.localizedDescription UTF8String]);
    }

    // Very dumb - just wanted the pipeline state to be created to get execution width
    if (tpr != 1) {
        // Validate that tpr, if more than the size of a simd group, is a multiple. 
        // TODO: Unfortunately, this must be done after we created the pipeline state, so it's in a weird place.
        if (tpr >= impl_->pipeline_state_with_bias.threadExecutionWidth && 
            tpr % impl_->pipeline_state_with_bias.threadExecutionWidth != 0
        ) {
            throw std::runtime_error("tpr must be less than or a multiple of the thread execution width (SIMD Group size)");
        }

        function_with_bias = tpr == impl_->pipeline_state_with_bias.threadExecutionWidth 
                 ? [library newFunctionWithName:@"linear_with_bias_1sgpr"]
                 : [library newFunctionWithName:@"linear_with_bias_nsgpr"];
        if (function_with_bias == nil) throw std::runtime_error("Function not found");
        impl_->pipeline_state_with_bias = [device newComputePipelineStateWithFunction:function_with_bias error:&error];
        if (impl_->pipeline_state_with_bias == nil) {
            throw std::runtime_error(std::string("Failed to create pipeline state: ") + [error.localizedDescription UTF8String]);
        }

        function_without_bias = tpr == impl_->pipeline_state_without_bias.threadExecutionWidth 
                 ? [library newFunctionWithName:@"linear_without_bias_1sgpr"]
                 : [library newFunctionWithName:@"linear_without_bias_nsgpr"];
        if (function_without_bias == nil) throw std::runtime_error("Function not found");
        impl_->pipeline_state_without_bias = [device newComputePipelineStateWithFunction:function_without_bias error:&error];
        if (impl_->pipeline_state_without_bias == nil) {
            throw std::runtime_error(std::string("Failed to create pipeline state: ") + [error.localizedDescription UTF8String]);
        }
    }
}
    
LinearKernel::~LinearKernel() = default;
LinearKernel::LinearKernel(LinearKernel&&) noexcept = default;
LinearKernel& LinearKernel::operator=(LinearKernel&&) noexcept = default;

// Without bias. 
// TODO: Fold in with the "with bias" implementation
MetalJob LinearKernel::submit_repeated(
    std::size_t repeats,
    const GpuTensor<float>& input,   // [B,S,I]
    const GpuTensor<float>& weight,  // [O,I]
    GpuTensor<float>& output         // [B,S,O]
) {
    if (in_progress()) throw std::runtime_error("Kernel is already in progress");
    if (repeats == 0) throw std::runtime_error("Invalid repeats");

    if (input.shape().rank() != 3) throw std::runtime_error("Invalid input shape, expected [batch_size, sequence_length, input_hidden_size]");
    if (weight.shape().rank() != 2) throw std::runtime_error("Invalid weights shape, expected [output_hidden_size, input_hidden_size]");
    if (output.shape().rank() != 3) throw std::runtime_error("Invalid output shape, expected [batch_size, sequence_length, output_hidden_size]");
    
    // Pull out shape information
    std::uint32_t O = checked_u32(weight.shape()[0], "linear_output_hidden_size"); // output_hidden_size
    std::uint32_t I = checked_u32(weight.shape()[1], "linear_input_hidden_size"); // input_hidden_size
    std::uint32_t B = checked_u32(input.shape()[0], "linear_batch_size"); // batch_size
    std::uint32_t S = checked_u32(input.shape()[1], "linear_sequence_length"); // sequence_length
    std::uint32_t n_input = checked_multiply(B, S);

    if (n_input == 0) throw std::runtime_error("Invalid input size 0");
    if (I == 0) throw std::runtime_error("Invalid hidden size");
    if (O == 0) throw std::runtime_error("Invalid hidden size");

    if (n_input * I != input.numel()) throw std::runtime_error("Invalid input size. Expected " + std::to_string(n_input * I) + " but got " + std::to_string(input.numel()));
    if (n_input * O != output.numel()) throw std::runtime_error("Invalid output size. Expected " + std::to_string(n_input * O) + " but got " + std::to_string(output.numel()));
    if (output.shape()[0] != B || output.shape()[1] != S || output.shape()[2] != O) {
        throw std::runtime_error("Invalid output shape. Expected [batch_size, sequence_length, output_hidden_size]");
    }

    id<MTLCommandQueue> commandQueue = (__bridge id<MTLCommandQueue>)impl_->metalContext.command_queue_handle();
    id<MTLCommandBuffer> commandBuffer = [commandQueue commandBuffer];
    if (commandBuffer == nil) throw std::runtime_error("Could not create command buffer");
    id<MTLComputeCommandEncoder> computeEncoder = [commandBuffer computeCommandEncoder];
    if (computeEncoder == nil) throw std::runtime_error("Could not create compute command encoder");

    // Encode command
    [computeEncoder setComputePipelineState:impl_->pipeline_state_without_bias];
    [computeEncoder setBuffer:(__bridge id<MTLBuffer>)input.buffer_handle()  offset:input.byte_offset_  atIndex:0];
    [computeEncoder setBuffer:(__bridge id<MTLBuffer>)weight.buffer_handle() offset:weight.byte_offset_ atIndex:1];
    [computeEncoder setBuffer:(__bridge id<MTLBuffer>)output.buffer_handle() offset:output.byte_offset_ atIndex:2];
    [computeEncoder setBytes:&I length:sizeof(uint) atIndex:3];
    [computeEncoder setBytes:&O length:sizeof(uint) atIndex:4];
    [computeEncoder setBytes:&S length:sizeof(uint) atIndex:5];

    if (impl_->tpr == 1) {
        // Naive - just one thread per output row
        MTLSize gridSize = MTLSizeMake(O, n_input, 1);
        NSUInteger upperBoundWidth = impl_->tptg > gridSize.width ? gridSize.width : impl_->tptg;
        MTLSize threadGroupSize = MTLSizeMake(upperBoundWidth, 1, 1);
        for (std::size_t i = 0; i < repeats; ++i) {
            [computeEncoder dispatchThreads:gridSize threadsPerThreadgroup:threadGroupSize];
        }
    } else if (impl_->tpr == impl_->pipeline_state_without_bias.threadExecutionWidth) {
        // 1 SIMD group per row, rptg rows per threadgroup
        NSUInteger rptg = impl_->tptg / impl_->tpr;
        MTLSize gridSize = MTLSizeMake(impl_->tpr, O, n_input);
        MTLSize threadGroupSize = MTLSizeMake(impl_->tpr, rptg, 1);
        for (std::size_t i = 0; i < repeats; ++i) {
            [computeEncoder dispatchThreads:gridSize threadsPerThreadgroup:threadGroupSize];
        }
    } else if (impl_->tpr > impl_->pipeline_state_without_bias.threadExecutionWidth) {
        // N SIMD group per row, 1 row per threadgroup
        NSUInteger sgptg = impl_->tpr / impl_->pipeline_state_without_bias.threadExecutionWidth;
        MTLSize gridSize = MTLSizeMake(impl_->tpr, O, n_input);
        MTLSize threadGroupSize = MTLSizeMake(impl_->tpr, 1, 1);
        [computeEncoder setBytes:&sgptg length:sizeof(uint) atIndex:6];
        for (std::size_t i = 0; i < repeats; ++i) {
            [computeEncoder dispatchThreads:gridSize threadsPerThreadgroup:threadGroupSize];
        }
    } else {
        // We don't support <32 per row, nor multiple simd groups and rows per threadgroup,
        // only allow eitehr multi simd per row or multi row per threadgroup with 1 simd group per row.
        throw std::runtime_error("Not implemented - tpr should be a multiple of the thread execution width");
    }
    [computeEncoder endEncoding];
    [commandBuffer commit];
    impl_->lastCommandBuffer = commandBuffer;
    return llmetal::MetalJob((__bridge void*) commandBuffer);
}

MetalJob LinearKernel::submit(
    const GpuTensor<float>& x,       // [B,S,I]
    const GpuTensor<float>& weight,  // [O,I]
    GpuTensor<float>& y
) {
    return submit_repeated(1, x, weight, y);
}

// With bias
MetalJob LinearKernel::submit_repeated(
    std::size_t repeats,
    const GpuTensor<float>& input,   // [B,S,I]
    const GpuTensor<float>& weight,  // [O,I]
    const GpuTensor<float>& bias,    // [O]
    GpuTensor<float>& output         // [B,S,O]
) {
    if (in_progress()) throw std::runtime_error("Kernel is already in progress");
    if (repeats == 0) throw std::runtime_error("Invalid repeats");

    if (input.shape().rank() != 3) throw std::runtime_error("Invalid input shape, expected [batch_size, sequence_length, input_hidden_size]");
    if (weight.shape().rank() != 2) throw std::runtime_error("Invalid weights shape, expected [output_hidden_size, input_hidden_size]");
    if (bias.shape().rank() != 1) throw std::runtime_error("Invalid bias shape, expected [output_hidden_size]");
    if (output.shape().rank() != 3) throw std::runtime_error("Invalid output shape, expected [batch_size, sequence_length, output_hidden_size]");
    
    // Pull out shape information
    std::uint32_t O = checked_u32(weight.shape()[0], "linear_output_hidden_size"); // output_hidden_size
    std::uint32_t I = checked_u32(weight.shape()[1], "linear_input_hidden_size"); // input_hidden_size
    std::uint32_t B = checked_u32(input.shape()[0], "linear_batch_size"); // batch_size
    std::uint32_t S = checked_u32(input.shape()[1], "linear_sequence_length"); // sequence_length
    std::uint32_t n_input = checked_multiply(B, S);

    if (n_input == 0) throw std::runtime_error("Invalid input size 0");
    if (I == 0) throw std::runtime_error("Invalid input hidden size");
    if (O == 0) throw std::runtime_error("Invalid output hidden size");

    if (n_input * I != input.numel()) throw std::runtime_error("Invalid input size. Expected " + std::to_string(n_input * I) + " but got " + std::to_string(input.numel()));
    if (n_input * O != output.numel()) throw std::runtime_error("Invalid output size. Expected " + std::to_string(n_input * O) + " but got " + std::to_string(output.numel()));
    if (bias.shape()[0] != O) throw std::runtime_error("Invalid bias shape. Expected [output_hidden_size]");
    if (output.shape()[0] != B || output.shape()[1] != S || output.shape()[2] != O) {
        throw std::runtime_error("Invalid output shape. Expected [batch_size, sequence_length, output_hidden_size]");
    }

    id<MTLCommandQueue> commandQueue = (__bridge id<MTLCommandQueue>)impl_->metalContext.command_queue_handle();
    id<MTLCommandBuffer> commandBuffer = [commandQueue commandBuffer];
    if (commandBuffer == nil) throw std::runtime_error("Could not create command buffer");
    id<MTLComputeCommandEncoder> computeEncoder = [commandBuffer computeCommandEncoder];
    if (computeEncoder == nil) throw std::runtime_error("Could not create compute command encoder");

    // Encode command
    [computeEncoder setComputePipelineState:impl_->pipeline_state_with_bias];
    [computeEncoder setBuffer:(__bridge id<MTLBuffer>)input.buffer_handle()  offset:input.byte_offset_  atIndex:0];
    [computeEncoder setBuffer:(__bridge id<MTLBuffer>)weight.buffer_handle() offset:weight.byte_offset_ atIndex:1];
    [computeEncoder setBuffer:(__bridge id<MTLBuffer>)bias.buffer_handle()   offset:bias.byte_offset_   atIndex:2];
    [computeEncoder setBuffer:(__bridge id<MTLBuffer>)output.buffer_handle() offset:output.byte_offset_ atIndex:3];
    [computeEncoder setBytes:&I length:sizeof(uint) atIndex:4];
    [computeEncoder setBytes:&O length:sizeof(uint) atIndex:5];
    [computeEncoder setBytes:&S length:sizeof(uint) atIndex:6];

    if (impl_->tpr == 1) {
        // Naive - just one thread per output row
        MTLSize gridSize = MTLSizeMake(O, n_input, 1);
        NSUInteger upperBoundWidth = impl_->tptg > gridSize.width ? gridSize.width : impl_->tptg;
        MTLSize threadGroupSize = MTLSizeMake(upperBoundWidth, 1, 1);
        for (std::size_t i = 0; i < repeats; ++i) {
            [computeEncoder dispatchThreads:gridSize threadsPerThreadgroup:threadGroupSize];
        }
    } else if (impl_->tpr == impl_->pipeline_state_with_bias.threadExecutionWidth) {
        // 1 SIMD group per row, rptg rows per threadgroup
        NSUInteger rptg = impl_->tptg / impl_->tpr;
        MTLSize gridSize = MTLSizeMake(impl_->tpr, O, n_input);
        MTLSize threadGroupSize = MTLSizeMake(impl_->tpr, rptg, 1);
        for (std::size_t i = 0; i < repeats; ++i) {
            [computeEncoder dispatchThreads:gridSize threadsPerThreadgroup:threadGroupSize];
        }
    } else if (impl_->tpr > impl_->pipeline_state_with_bias.threadExecutionWidth) {
        // N SIMD group per row, 1 row per threadgroup
        NSUInteger sgptg = impl_->tpr / impl_->pipeline_state_with_bias.threadExecutionWidth;
        MTLSize gridSize = MTLSizeMake(impl_->tpr, O, n_input);
        MTLSize threadGroupSize = MTLSizeMake(impl_->tpr, 1, 1);
        [computeEncoder setBytes:&sgptg length:sizeof(uint) atIndex:7];
        for (std::size_t i = 0; i < repeats; ++i) {
            [computeEncoder dispatchThreads:gridSize threadsPerThreadgroup:threadGroupSize];
        }
    } else {
        // We don't support <32 per row, nor multiple simd groups and rows per threadgroup,
        // only allow eitehr multi simd per row or multi row per threadgroup with 1 simd group per row.
        throw std::runtime_error("Not implemented - tpr should be a multiple of the thread execution width");
    }
    [computeEncoder endEncoding];
    [commandBuffer commit];
    impl_->lastCommandBuffer = commandBuffer;
    return llmetal::MetalJob((__bridge void*) commandBuffer);
}

MetalJob LinearKernel::submit(
    const GpuTensor<float>& x,       // [B,S,I]
    const GpuTensor<float>& weight,  // [O,I]
    const GpuTensor<float>& bias,    // [O]
    GpuTensor<float>& y
) {
    return submit_repeated(1, x, weight, bias, y);
}

bool LinearKernel::in_progress() const noexcept {
    if (impl_->lastCommandBuffer == nil) return false;
    const auto status = [impl_->lastCommandBuffer status];
    return status != MTLCommandBufferStatusCompleted && 
           status != MTLCommandBufferStatusError;
}

} // namespace llmetal
