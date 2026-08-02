#include <cstring>
#include <format>
#include <iostream>
#include <llmetal/metal_context.hpp>
#include <llmetal/vector_add.hpp>

#include <Foundation/Foundation.h>
#include <Metal/Metal.h>

namespace llmetal {

class VectorAddKernel::Impl {
public:
    explicit Impl(MetalContext& context)
        : metalContext(context) {}
    
    MetalContext& metalContext;
    id<MTLComputePipelineState> pipeline_state = nil;
};

VectorAddKernel::VectorAddKernel(MetalContext& context)
    : impl_(std::make_unique<Impl>(context)) {
    NSError* error = nil;
    
    id<MTLDevice> device = (__bridge id<MTLDevice>)context.device_handle();
    if (device == nil) throw std::runtime_error("Device not found");

    // We need to manually get the library from the shaders folder
    NSURL* exeURL = [[NSBundle mainBundle] executableURL];
    NSURL* libURL = [
        [exeURL URLByDeletingLastPathComponent] URLByAppendingPathComponent:@"shaders/vector_add.metallib"
    ];  
    id<MTLLibrary> library = [device newLibraryWithURL:libURL error:&error];
    if (library == nil) {
        throw std::runtime_error(
            std::string("Library error: ") + [error.localizedDescription UTF8String]
        );
    }

    // From `shaders/vector_add.metal`
    id<MTLFunction> function = [library newFunctionWithName:@"my_add_arrays"];
    if (function == nil) throw std::runtime_error("Function not found");

    impl_->pipeline_state = [device newComputePipelineStateWithFunction: function error:&error];
    if (impl_->pipeline_state == nil) { 
        throw std::runtime_error(
            std::string("Failed to create pipeline state: ") 
            + [error.localizedDescription UTF8String]
        );
    }
}

VectorAddKernel::~VectorAddKernel() = default;
VectorAddKernel::VectorAddKernel(VectorAddKernel&&) noexcept = default;
VectorAddKernel& VectorAddKernel::operator=(VectorAddKernel&&) noexcept = default;

void VectorAddKernel::encodeAddCommand(
    // MTLComputeCommandEncoder
    void* _computeEncoder,
    // MTLBuffer
    void* _bufferLhs,
    // MTLBuffer
    void* _bufferRhs,
    // MTLBuffer
    void* _bufferOutput,
    unsigned int elementCount
) {
    id<MTLComputeCommandEncoder> computeEncoder = (__bridge id<MTLComputeCommandEncoder>)_computeEncoder;
    id<MTLBuffer> bufferLhs = (__bridge id<MTLBuffer>)_bufferLhs;
    id<MTLBuffer> bufferRhs = (__bridge id<MTLBuffer>)_bufferRhs;
    id<MTLBuffer> bufferOutput = (__bridge id<MTLBuffer>)_bufferOutput;

    // Encode the compute command
    [computeEncoder setComputePipelineState:impl_->pipeline_state];
    [computeEncoder setBuffer:bufferLhs offset:0 atIndex:0];
    [computeEncoder setBuffer:bufferRhs offset:0 atIndex:1];
    [computeEncoder setBuffer:bufferOutput offset:0 atIndex:2];

    MTLSize gridSize = MTLSizeMake(elementCount, 1, 1);

    NSUInteger upperBound = impl_->pipeline_state.maxTotalThreadsPerThreadgroup;
    if (upperBound > gridSize.width) {
        upperBound = gridSize.width;
    }
    MTLSize threadGroupSize = MTLSizeMake(upperBound, 1, 1);

    [computeEncoder dispatchThreads:gridSize threadsPerThreadgroup:threadGroupSize];
    [computeEncoder endEncoding];
}

void VectorAddKernel::run(
    std::span<const float> lhs,
    std::span<const float> rhs,
    std::span<float> output
) {
    const NSUInteger bufferSize = lhs.size() * sizeof(float);
    if (lhs.size() != rhs.size() || lhs.size() != output.size()) {
        throw std::runtime_error("Invalid input sizes");
    }

    id<MTLDevice> device = (__bridge id<MTLDevice>)impl_->metalContext.device_handle();
    if (device == nil) throw std::runtime_error("Device not found");
    id<MTLCommandQueue> commandQueue = (__bridge id<MTLCommandQueue>)impl_->metalContext.command_queue_handle();
    if (commandQueue == nil) throw std::runtime_error("Command queue not found");
    id<MTLCommandBuffer> commandBuffer = [commandQueue commandBuffer];
    if (commandBuffer == nil) throw std::runtime_error("Could not create command buffer");
    id<MTLComputeCommandEncoder> computeEncoder = [commandBuffer computeCommandEncoder];
    if (computeEncoder == nil) throw std::runtime_error("Could not create compute command encoder");

    // Prepare data
    id<MTLBuffer> bufferLhs = [device newBufferWithLength:bufferSize options:MTLResourceStorageModeShared];
    id<MTLBuffer> bufferRhs = [device newBufferWithLength:bufferSize options:MTLResourceStorageModeShared];
    id<MTLBuffer> bufferOutput = [device newBufferWithLength:bufferSize options:MTLResourceStorageModeShared];
    void* lhsData = (void*) bufferLhs.contents;
    void* rhsData = (void*) bufferRhs.contents;
    std::memcpy(lhsData, lhs.data(), bufferSize);
    std::memcpy(rhsData, rhs.data(), bufferSize);

    // Encode command
    encodeAddCommand(computeEncoder, bufferLhs, bufferRhs, bufferOutput, lhs.size());

    // Execute command
    [commandBuffer commit];
    [commandBuffer waitUntilCompleted];

    // Retrieve results
    float* result = static_cast<float*>([bufferOutput contents]);
    std::copy(result, result + output.size(), output.data());

    // Print results
    std::cout << "Input LHS: ";
    for (unsigned long index = 0; index < lhs.size(); index++) {
        std::cout << lhs[index] << " ";
    }
    std::cout << std::endl << "Input RHS: ";
    for (unsigned long index = 0; index < rhs.size(); index++) {
        std::cout << rhs[index] << " ";
    }
    std::cout << std::endl << "Output: ";
    for (unsigned long index = 0; index < lhs.size(); index++) {
        std::cout << result[index] << " ";
    }
    std::cout << std::endl;
}

} //namespace llmetal