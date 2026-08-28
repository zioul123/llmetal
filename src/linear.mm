#include "llmetal/tensor.hpp"
#include "llmetal/linear.hpp"
#include "llmetal/metal_context.hpp"
#include "llmetal/metal_job.hpp"

#include <Foundation/Foundation.h>
#include <Metal/Metal.h>
#include <objc/NSObjCRuntime.h>
#include <stdexcept>
#include <string>

namespace llmetal {

class LinearKernel::Impl {
public:
    explicit Impl(MetalContext& context, 
                  std::uint32_t tptg,
                  std::uint32_t tpr,
                  std::uint32_t rpt,
                  std::uint32_t npt)
        : metalContext(context), tptg(tptg), tpr(tpr), rpt(rpt), npt(npt) {};
    MetalContext& metalContext;
    std::uint32_t tptg; // threads per threadgroup
    std::uint32_t tpr;  // threads per row - only applicable for 1D output row tiled
    std::uint32_t rpt;  // rows per row group (extra arithmetic intensity)
    std::uint32_t npt;  // batch/sequence per row group (extra arithmetic intensity)
private:
    id<MTLComputePipelineState> pipeline_state_with_bias = nil;
    id<MTLComputePipelineState> pipeline_state_without_bias = nil;
    id<MTLCommandBuffer> lastCommandBuffer = nil;
    static id<MTLFunction> specialize(
        id<MTLLibrary> library, NSString* name, bool has_bias
    ) {
        NSError* error = nil;
        MTLFunctionConstantValues* cv = [MTLFunctionConstantValues new];
        [cv setConstantValue:&has_bias type:MTLDataTypeBool atIndex:0];
        id<MTLFunction> fn = [library newFunctionWithName:name 
                                           constantValues:cv 
                                                    error:&error];
        if (fn == nil) throw std::runtime_error(
            std::string("Function ") + [name UTF8String] + ": " + 
            [error.localizedDescription UTF8String]
        );
        return fn;
    }
friend class LinearKernel;
};


LinearKernel::LinearKernel(MetalContext& context,
                           std::uint32_t tptg,
                           std::uint32_t tpr,
                           std::uint32_t rpt,
                           std::uint32_t npt)
    : impl_(std::make_unique<Impl>(context, tptg, tpr, rpt, npt)) {
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
    if (rpt != 1 && npt != 1 &&  // 2D tiled -> only the two arrangements below are allowed
        !(rpt == 4 && npt == 4 && tptg == 256) && !(rpt == 8 && npt == 8 && tptg == 256)
    ) {
        throw std::runtime_error("Either rpt or npt must be 1, or (rpt, npt) must be (4, 4) or (8, 8), and tptg must be 256");
    }
    if (npt != 1 && tpr != 1) {
        throw std::runtime_error("If npt != 1, only naive (tpr == 1) is implemented");
    }
    if (rpt != 1 && rpt != 2 && rpt != 4 && rpt != 8) {  // too verbose, use array
        throw std::runtime_error("rpt must be 1, 2, 4, or 8");
    }

    // Just a throwaway so we can get the thread execution width 
    NSUInteger threadExecutionWidth; 
    {
        id<MTLFunction> throwaway_function = LinearKernel::Impl::specialize(
            library, @"linear_naive", false
        );
        auto throwaway_pipeline = [device newComputePipelineStateWithFunction:throwaway_function error:&error];
        if (throwaway_pipeline == nil) throw std::runtime_error(std::string("Failed to create pipeline state: ") + [error.localizedDescription UTF8String]);
        threadExecutionWidth = throwaway_pipeline.threadExecutionWidth;
    }
    std::string name_str = rpt != 1 && npt != 1 ? "linear_2d_tiled_" + std::to_string(rpt) + "x" + std::to_string(npt)
                         : tpr == 1 ? "linear_naive"
                         : tpr == threadExecutionWidth ? "linear_1sgpr"
                         : "linear_nsgpr";
    if (rpt != 1 && npt != 1) {}
    else if (rpt != 1) name_str = name_str + "_rprgx" + std::to_string(rpt);
    else if (npt != 1) name_str = name_str + "_bsprgx" + std::to_string(npt);
    NSString* name = @(name_str.c_str());
    id<MTLFunction> function_with_bias = LinearKernel::Impl::specialize(
        library, name, true
    );
    impl_->pipeline_state_with_bias = [device newComputePipelineStateWithFunction:function_with_bias error:&error];
    if (impl_->pipeline_state_with_bias == nil) {
        throw std::runtime_error(std::string("Failed to create pipeline state: ") + 
                                 [error.localizedDescription UTF8String]);
    }

    id<MTLFunction> function_without_bias = LinearKernel::Impl::specialize(
        library, name, false
    );;
    impl_->pipeline_state_without_bias = [device newComputePipelineStateWithFunction:function_without_bias error:&error];
    if (impl_->pipeline_state_without_bias == nil) {
        throw std::runtime_error(std::string("Failed to create pipeline state: ") 
                                 + [error.localizedDescription UTF8String]);
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
    const GpuTensor<float>* bias,    // [O], nullptr = no bias
    GpuTensor<float>& output         // [B,S,O]
) {
    // Sanity checks
    if (in_progress()) throw std::runtime_error("Kernel is already in progress");
    if (repeats == 0) throw std::runtime_error("Invalid repeats");

    // Shape dimension validation
    if (input.shape().rank() != 3) throw std::runtime_error("Invalid input shape, expected [batch_size, sequence_length, input_hidden_size]");
    if (weight.shape().rank() != 2) throw std::runtime_error("Invalid weights shape, expected [output_hidden_size, input_hidden_size]");
    if (output.shape().rank() != 3) throw std::runtime_error("Invalid output shape, expected [batch_size, sequence_length, output_hidden_size]");
    if (bias && bias->shape().rank() != 1) throw std::runtime_error("Invalid bias shape, expected [output_hidden_size]");
    
    // Pull out shape information
    std::uint32_t O = checked_u32(weight.shape()[0], "linear_output_hidden_size"); // output_hidden_size
    std::uint32_t I = checked_u32(weight.shape()[1], "linear_input_hidden_size"); // input_hidden_size
    std::uint32_t B = checked_u32(input.shape()[0], "linear_batch_size"); // batch_size
    std::uint32_t S = checked_u32(input.shape()[1], "linear_sequence_length"); // sequence_length
    std::uint32_t n_input = checked_multiply(B, S);

    // Shape validation
    if (n_input == 0) throw std::runtime_error("Invalid input size 0");
    if (I == 0) throw std::runtime_error("Invalid hidden size");
    if (O == 0) throw std::runtime_error("Invalid hidden size");
    if (n_input * I != input.numel()) throw std::runtime_error("Invalid input size. Expected " + std::to_string(n_input * I) + " but got " + std::to_string(input.numel()));
    if (n_input * O != output.numel()) throw std::runtime_error("Invalid output size. Expected " + std::to_string(n_input * O) + " but got " + std::to_string(output.numel()));
    if (output.shape()[0] != B || output.shape()[1] != S || output.shape()[2] != O) {
        throw std::runtime_error("Invalid output shape. Expected [batch_size, sequence_length, output_hidden_size]");
    }
    if (bias && bias->shape()[0] != O) throw std::runtime_error("Invalid bias shape. Expected [output_hidden_size]");

    // Retrieve encoding objects
    id<MTLCommandQueue> commandQueue = (__bridge id<MTLCommandQueue>)impl_->metalContext.command_queue_handle();
    id<MTLCommandBuffer> commandBuffer = [commandQueue commandBuffer];
    if (commandBuffer == nil) throw std::runtime_error("Could not create command buffer");
    id<MTLComputeCommandEncoder> computeEncoder = [commandBuffer computeCommandEncoder];
    if (computeEncoder == nil) throw std::runtime_error("Could not create compute command encoder");
    id<MTLComputePipelineState> pso = bias ? impl_->pipeline_state_with_bias : impl_->pipeline_state_without_bias;
    
    // Encode command
    [computeEncoder setComputePipelineState:pso];
    [computeEncoder setBuffer:(__bridge     id<MTLBuffer>)input.buffer_handle()  offset:input.byte_offset_  atIndex:0];
    [computeEncoder setBuffer:(__bridge     id<MTLBuffer>)weight.buffer_handle() offset:weight.byte_offset_ atIndex:1];
    if (bias) {
        [computeEncoder setBuffer:(__bridge id<MTLBuffer>)bias->buffer_handle()  offset:bias->byte_offset_   atIndex:2];    
    }
    [computeEncoder setBuffer:(__bridge id<MTLBuffer>)output.buffer_handle() offset:output.byte_offset_ atIndex:3];
    [computeEncoder setBytes:&I length:sizeof(uint) atIndex:4];
    [computeEncoder setBytes:&O length:sizeof(uint) atIndex:5];
    [computeEncoder setBytes:&S length:sizeof(uint) atIndex:6];

    // Dispatch specific kernel
    // 2D tile
    if (impl_->rpt != 1 && impl_->npt != 1) {
        std::uint32_t input_groups = (n_input + impl_->npt - 1) / impl_->npt;
        std::uint32_t rowGroups = (O + impl_->rpt - 1) / impl_->rpt;

        MTLSize gridSize = MTLSizeMake(rowGroups, input_groups, 1);
        MTLSize threadGroupSize = MTLSizeMake(16, 16, 1);
        [computeEncoder setBytes:&n_input length:sizeof(uint) atIndex:7];
        for (std::size_t i = 0; i < repeats; ++i) {
            [computeEncoder dispatchThreads:gridSize threadsPerThreadgroup:threadGroupSize];
        }

    }
    else if (impl_->tpr == 1) {
        // One row group (1, 2, 4 or 8 rows) managed by exactly 1 thread
        if (impl_->npt == 1) {
            // Just one thread per row group 
            std::uint32_t rowGroups = (O + impl_->rpt - 1) / impl_->rpt;
            MTLSize gridSize = MTLSizeMake(rowGroups, n_input, 1);
            NSUInteger upperBoundWidth = impl_->tptg > gridSize.width ? gridSize.width : impl_->tptg;
            MTLSize threadGroupSize = MTLSizeMake(upperBoundWidth, 1, 1);
            for (std::size_t i = 0; i < repeats; ++i) {
                [computeEncoder dispatchThreads:gridSize threadsPerThreadgroup:threadGroupSize];
            }
        } 
        // One row managed by exactly 1 thread, bsprg (1, 2, 4 or 8) input tokens at once
        else {
            // Just one thread per row, but multiple inputs in parallel
            std::uint32_t input_groups = (n_input + impl_->npt - 1) / impl_->npt;
            MTLSize gridSize = MTLSizeMake(O, input_groups, 1);
            NSUInteger upperBoundWidth = impl_->tptg > gridSize.width ? gridSize.width : impl_->tptg;
            MTLSize threadGroupSize = MTLSizeMake(upperBoundWidth, 1, 1);
            [computeEncoder setBytes:&n_input length:sizeof(uint) atIndex:7];
            for (std::size_t i = 0; i < repeats; ++i) {
                [computeEncoder dispatchThreads:gridSize threadsPerThreadgroup:threadGroupSize];
            }
        }
    } 
    // These two are 1D tiled by just output row
    else if (impl_->tpr == impl_->pipeline_state_without_bias.threadExecutionWidth) {
        // 1 SIMD group per row group, rgptg row groups per threadgroup
        std::uint32_t rowGroups = (O + impl_->rpt - 1) / impl_->rpt;
        NSUInteger rgptg = impl_->tptg / impl_->tpr;
        MTLSize gridSize = MTLSizeMake(impl_->tpr, rowGroups, n_input);
        NSUInteger upperBoundHeight = rgptg > gridSize.height ? gridSize.height : rgptg;
        MTLSize threadGroupSize = MTLSizeMake(impl_->tpr, upperBoundHeight, 1);
        for (std::size_t i = 0; i < repeats; ++i) {
            [computeEncoder dispatchThreads:gridSize threadsPerThreadgroup:threadGroupSize];
        }
    } else if (impl_->tpr > impl_->pipeline_state_without_bias.threadExecutionWidth) {
        // N SIMD group per row, 1 row per threadgroup
        std::uint32_t rowGroups = (O + impl_->rpt - 1) / impl_->rpt;
        std::uint32_t sgpr = impl_->tpr / impl_->pipeline_state_without_bias.threadExecutionWidth;
        MTLSize gridSize = MTLSizeMake(impl_->tpr, rowGroups, n_input);
        MTLSize threadGroupSize = MTLSizeMake(impl_->tpr, 1, 1);
        [computeEncoder setBytes:&sgpr length:sizeof(uint) atIndex:7];
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
    const GpuTensor<float>* bias,    // [O], nullptr = no bias
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
