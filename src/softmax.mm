#include "llmetal/tensor.hpp"
#include "llmetal/softmax.hpp"
#include "llmetal/metal_context.hpp"
#include "llmetal/metal_job.hpp"

#include <Foundation/Foundation.h>
#include <Metal/Metal.h>

namespace llmetal {

class SoftmaxKernel::Impl {
public:
    explicit Impl(MetalContext& context,
                  std::uint32_t tptg,
                  std::uint32_t tpr,
                  std::uint32_t rpt)
        : metalContext(context), tptg(tptg), tpr(tpr), rpt(rpt) {};
    MetalContext& metalContext;
    std::uint32_t tptg; // threads per threadgroup
    std::uint32_t tpr;  // threads per row
    std::uint32_t rpt;  // rows per row group (extra arithmetic intensity)
private:
    id<MTLComputePipelineState> pipeline_state = nil;
    id<MTLCommandBuffer> lastCommandBuffer = nil;
friend class SoftmaxKernel;
};

SoftmaxKernel::SoftmaxKernel(MetalContext& context,
                             std::uint32_t tptg,
                             std::uint32_t tpr,
                             std::uint32_t rpt)
    : impl_(std::make_unique<Impl>(context, tptg, tpr, rpt)) {
    NSError* error = nil;
    id<MTLDevice> device = (__bridge id<MTLDevice>)context.device_handle();

    // We need to manually get the library from the shaders folder
    NSURL* exeURL = [[NSBundle mainBundle] executableURL];
    NSURL* libURL = [
        [exeURL URLByDeletingLastPathComponent] URLByAppendingPathComponent:@"shaders/softmax.metallib"
    ];  
    id<MTLLibrary> library = [device newLibraryWithURL:libURL error:&error];
    if (library == nil) {
        throw std::runtime_error(
            std::string("Library error: ") + [error.localizedDescription UTF8String]
        );
    }

    // Just a throwaway so we can get the thread execution width 
    NSUInteger threadExecutionWidth; 
    {
        id<MTLFunction> throwaway_function = [library newFunctionWithName:@"softmax_naive"];
        auto throwaway_pipeline = [device newComputePipelineStateWithFunction:throwaway_function error:&error];
        if (throwaway_pipeline == nil) throw std::runtime_error(std::string("Failed to create pipeline state: ") + [error.localizedDescription UTF8String]);
        threadExecutionWidth = throwaway_pipeline.threadExecutionWidth;
    }

    if (tptg % tpr != 0) {
        throw std::runtime_error("tptg must be a multiple of tpr");
    }
    if (rpt != 1 && rpt != 2 && rpt != 4 && rpt != 8) {  // too verbose, use array
        throw std::runtime_error("rpt must be 1, 2, 4, or 8");
    }
    if (rpt == 1 && tpr != 1 && tpr % threadExecutionWidth != 0) {
        throw std::runtime_error("tpr must be a multiple of the thread execution width");   
    }

    std::string name_str = rpt == 1 && tpr == 1 ? "softmax_naive"
                         : rpt != 1 && tpr == 1 ? "softmax_" + std::to_string(rpt) + "rpt"
                         : rpt == 1 && tpr != 1 ? "softmax_" + std::to_string(tpr / threadExecutionWidth) + "sgpr"
                         : "softmax_2d_tiled_" + std::to_string(tpr / threadExecutionWidth) + "x" + std::to_string(rpt);

    id<MTLFunction> function = [library newFunctionWithName:@(name_str.c_str())];
    if (function == nil) throw std::runtime_error("Function not found");

    impl_->pipeline_state = [device newComputePipelineStateWithFunction:function error:&error];
    if (impl_->pipeline_state == nil) {
        throw std::runtime_error(std::string("Failed to create pipeline state: ") 
                                 + [error.localizedDescription UTF8String]);
    }
}
    
SoftmaxKernel::~SoftmaxKernel() = default;
SoftmaxKernel::SoftmaxKernel(SoftmaxKernel&&) noexcept = default;
SoftmaxKernel& SoftmaxKernel::operator=(SoftmaxKernel&&) noexcept = default;


MetalJob SoftmaxKernel::submit_repeated(
    std::size_t repeats,
    const GpuTensor<float>& logits,        // [rows,C]
    const GpuTensor<std::uint32_t>& valid, // [rows]
    GpuTensor<float>& output               // [rows,C]
) {
    if (repeats == 0) throw std::runtime_error("Invalid repeats");
    if (in_progress()) throw std::runtime_error("Kernel is already in progress");

    if (logits.shape().rank() != 2) throw std::invalid_argument("Invalid logits shape rank");
    if (valid.shape().rank() != 1) throw std::invalid_argument("Invalid valid shape rank");
    if (output.shape().rank() != 2) throw std::invalid_argument("Invalid output shape rank");

    // Pull out shape information
    std::uint32_t rows = checked_u32(logits.shape()[0], "softmax_rows");
    std::uint32_t seq_length = checked_u32(logits.shape()[1], "softmax_seq_length");
    if (valid.shape()[0] != rows) throw std::invalid_argument("Invalid valid shape");
    if (output.numel() != logits.numel()) throw std::invalid_argument("Invalid output shape - expected same size as logits");
    
    id<MTLCommandQueue> commandQueue = (__bridge id<MTLCommandQueue>)impl_->metalContext.command_queue_handle();

    // Create command buffer and encoder
    id<MTLCommandBuffer> commandBuffer = [commandQueue commandBuffer];
    if (commandBuffer == nil) throw std::runtime_error("Could not create command buffer");
    id<MTLComputeCommandEncoder> computeEncoder = [commandBuffer computeCommandEncoder];
    if (computeEncoder == nil) throw std::runtime_error("Could not create compute command encoder");

    // Encode command
    [computeEncoder setComputePipelineState:impl_->pipeline_state];
    [computeEncoder setBuffer:(__bridge id<MTLBuffer>)logits.buffer_handle() offset:logits.byte_offset_ atIndex:0];
    [computeEncoder setBuffer:(__bridge id<MTLBuffer>)valid.buffer_handle()  offset:valid.byte_offset_  atIndex:1];
    [computeEncoder setBuffer:(__bridge id<MTLBuffer>)output.buffer_handle() offset:output.byte_offset_ atIndex:2];
    [computeEncoder setBytes:&seq_length length:sizeof(uint) atIndex:3];

    if (impl_->tpr == 1 && impl_->rpt == 1) {
        // Naive - just one thread per row, one row per thread
        MTLSize gridSize = MTLSizeMake(rows, 1, 1);
        NSUInteger upperBound = impl_->tptg;
        upperBound = (upperBound > gridSize.width) ? gridSize.width : upperBound;
        MTLSize threadGroupSize = MTLSizeMake(upperBound, 1, 1);
        for (std::size_t i = 0; i < repeats; ++i) {
            [computeEncoder dispatchThreads:gridSize threadsPerThreadgroup:threadGroupSize];
        }
    } else if (impl_->tpr == impl_->pipeline_state.threadExecutionWidth) {
        // // 1 SIMD group per row, rptg rows per threadgroup
        // NSUInteger rptg = impl_->tptg / impl_->tpr;
        // MTLSize gridSize = MTLSizeMake(impl_->tpr, n_input, 1);
        // MTLSize threadGroupSize = MTLSizeMake(impl_->tpr, rptg, 1);
        // for (std::size_t i = 0; i < repeats; ++i) {
        //     [computeEncoder dispatchThreads:gridSize threadsPerThreadgroup:threadGroupSize];
        // }
        throw std::runtime_error("Not implemented 1 - tpr should be a multiple of the thread execution width");
    } else if (impl_->tpr > impl_->pipeline_state.threadExecutionWidth) {
        // // N SIMD group per row, 1 row per threadgroup
        // NSUInteger sgptg = impl_->tpr / impl_->pipeline_state.threadExecutionWidth;
        // MTLSize gridSize = MTLSizeMake(impl_->tpr, n_input, 1);
        // MTLSize threadGroupSize = MTLSizeMake(impl_->tpr, 1, 1);
        // [computeEncoder setBytes:&sgptg length:sizeof(uint) atIndex:5];
        // for (std::size_t i = 0; i < repeats; ++i) {
        //     [computeEncoder dispatchThreads:gridSize threadsPerThreadgroup:threadGroupSize];
        // }
        throw std::runtime_error("Not implemented 2 - tpr should be a multiple of the thread execution width");
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

MetalJob SoftmaxKernel::submit(
    const GpuTensor<float>& logits,        // [rows,C]
    const GpuTensor<std::uint32_t>& valid, // [rows]
    GpuTensor<float>& probs                // [rows,C]
) {
    return submit_repeated(1, logits, valid, probs);
}

bool SoftmaxKernel::in_progress() const noexcept {
    if (impl_->lastCommandBuffer == nil) return false;
    const auto status = [impl_->lastCommandBuffer status];
    return status != MTLCommandBufferStatusCompleted && 
           status != MTLCommandBufferStatusError;
}


} // namespace llmetal
