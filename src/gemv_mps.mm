#include <iostream>
#include <stdexcept>
#include <cstddef>
#include <cstring>
#include <string>
#include <algorithm>

#include <llmetal/metal_context.hpp>
#include <llmetal/gemv_mps.hpp>

#include <Foundation/Foundation.h>
#include <Metal/Metal.h>
#include <MetalPerformanceShaders/MetalPerformanceShaders.h>


namespace llmetal {

class GemvMpsKernel::Impl {
public:
    explicit Impl(MetalContext& context): metalContext(context) {}
    MetalContext& metalContext;
private:
    id<MTLCommandBuffer> lastCommandBuffer = nil;
    MPSMatrixVectorMultiplication* mpsKernel = nil;
    MPSMatrix* matrix = nil;
    MPSVector* vector = nil;
    MPSVector* output = nil;
    std::size_t capacityMatrix = 0;
    std::size_t capacityVector = 0;
    std::size_t capacityOutput = 0;
    id<MTLBuffer> bufferMatrix = nil;
    id<MTLBuffer> bufferVector = nil;
    id<MTLBuffer> bufferOutput = nil;
    Shape gemvShape = {0, 0};
friend class GemvMpsKernel;
};

GemvMpsKernel::GemvMpsKernel(MetalContext& context)
    : impl_(std::make_unique<Impl>(context)) {}
GemvMpsKernel::~GemvMpsKernel() = default;
GemvMpsKernel::GemvMpsKernel(GemvMpsKernel&&) noexcept = default;
GemvMpsKernel& GemvMpsKernel::operator=(GemvMpsKernel&&) noexcept = default;

void GemvMpsKernel::prepare(Shape shape) {
    std::uint32_t rows = checked_u32(shape[0], "rows");
    std::uint32_t cols = checked_u32(shape[1], "cols");
    if (in_progress()) throw std::runtime_error("Kernel is already in progress");
    if (cols == 0 || rows == 0) throw std::runtime_error("Invalid gemv shape");

    const NSUInteger rowBytes = cols * sizeof(float);
    const NSUInteger colBytes = rows * sizeof(float);
    const NSUInteger matrixBytes = rows * rowBytes;
    std::size_t matElements = cols * rows;
    id<MTLDevice> device = (__bridge id<MTLDevice>)impl_->metalContext.device_handle();

    if (matElements > impl_->capacityMatrix) {
        id<MTLBuffer> bufferMatrix =
            [device newBufferWithLength:matrixBytes
                                options:MTLResourceStorageModeShared];
        if (bufferMatrix == nil ) throw std::runtime_error("Failed to create matrix buffer");
        impl_->bufferMatrix = bufferMatrix;
        impl_->capacityMatrix = matElements;

    }
    if (cols > impl_->capacityVector) {
        id<MTLBuffer> bufferVector =
            [device newBufferWithLength:rowBytes
                                options: MTLResourceStorageModeShared];
        if (bufferVector == nil ) throw std::runtime_error("Failed to create vector buffer");
        impl_->bufferVector = bufferVector;
        impl_->capacityVector = cols;

    }

    if (rows > impl_->capacityOutput) {
        id<MTLBuffer> bufferOutput =
            [device newBufferWithLength: colBytes 
                                options: MTLResourceStorageModeShared];
        if (bufferOutput == nil) throw std::runtime_error("Failed to create buffers");
        impl_->bufferOutput = bufferOutput;
        impl_->capacityOutput = rows;

    }

    MPSMatrixDescriptor* matrixDesc = 
        [MPSMatrixDescriptor matrixDescriptorWithRows:rows
                                                columns:cols 
                                                rowBytes:rowBytes 
                                                dataType:MPSDataTypeFloat32 ];
    impl_->matrix  = [[MPSMatrix alloc] initWithBuffer:impl_->bufferMatrix 
                                            descriptor:matrixDesc];
    MPSVectorDescriptor* vectorDesc = 
        [MPSVectorDescriptor vectorDescriptorWithLength:cols 
                                                dataType:MPSDataTypeFloat32];
    impl_->vector = [[MPSVector alloc] initWithBuffer:impl_->bufferVector 
                                            descriptor:vectorDesc];
    MPSVectorDescriptor* outputDesc = 
        [MPSVectorDescriptor vectorDescriptorWithLength:rows 
                                                dataType:MPSDataTypeFloat32];
    impl_->output = [[MPSVector alloc] initWithBuffer:impl_->bufferOutput 
                                            descriptor:outputDesc];

    impl_->mpsKernel = 
        [[MPSMatrixVectorMultiplication alloc] initWithDevice:device
                                                         rows:rows
                                                      columns:cols];
    if (!impl_->mpsKernel) throw std::runtime_error("Failed to create MPS GEMV kernel");
    impl_->gemvShape = shape;
}

void GemvMpsKernel::upload_matrix(std::span<const float> matrix) {
    if (in_progress()) throw std::runtime_error("Kernel is already in progress");
    if (matrix.size() != impl_->gemvShape[1] * impl_->gemvShape[0]) {
        throw std::runtime_error(
            "Invalid matrix size. Expected: "
            + std::to_string(impl_->gemvShape[1] * impl_->gemvShape[0])
            + ", got: " + std::to_string(matrix.size())
        );
    }
    std::memcpy([impl_->bufferMatrix contents], matrix.data(), matrix.size() * sizeof(float));
}

void GemvMpsKernel::upload_vector(std::span<const float> vector) {
    if (in_progress()) throw std::runtime_error("Kernel is already in progress");
    if (vector.size() != impl_->gemvShape[1]) {
        throw std::runtime_error(
            "Invalid vector size. Expected: "
            + std::to_string(impl_->gemvShape[1])
            + ", got: " + std::to_string(vector.size())
        );
    }
    std::memcpy([impl_->bufferVector contents], vector.data(), vector.size() * sizeof(float));
}

llmetal::MetalJob GemvMpsKernel::submit_repeated(std::size_t repeats) {
    if (repeats == 0) throw std::runtime_error("Invalid repeats");
    if (in_progress()) throw std::runtime_error("Kernel is already in progress");
    if (impl_->gemvShape[1] == 0 || impl_->gemvShape[0] == 0) throw std::runtime_error("Invalid gemv shape");

    id<MTLCommandQueue> commandQueue = (__bridge id<MTLCommandQueue>)impl_->metalContext.command_queue_handle();

    // Create command buffer
    id<MTLCommandBuffer> commandBuffer = [commandQueue commandBuffer];
    if (commandBuffer == nil) throw std::runtime_error("Could not create command buffer");
    
    for (std::size_t i = 0; i < repeats; ++i) {
        [impl_->mpsKernel encodeToCommandBuffer:commandBuffer 
                                    inputMatrix:impl_->matrix 
                                    inputVector:impl_->vector
                                   resultVector:impl_->output];
    }
    
    // Execute command
    [commandBuffer commit];
    impl_->lastCommandBuffer = commandBuffer;
    return llmetal::MetalJob((__bridge void*) commandBuffer);
}

llmetal::MetalJob GemvMpsKernel::submit() {
    return submit_repeated(1);
}

void GemvMpsKernel::download(std::span<float> output) {
    if (in_progress()) throw std::runtime_error("Kernel is already in progress");
    if (output.size() < impl_->gemvShape[0]) {
        throw std::runtime_error(
            "Invalid output size. Expected: "
            + std::to_string(impl_->gemvShape[0])
            + ", got: " + std::to_string(output.size())
        );
    }
    std::memcpy(output.data(), [impl_->bufferOutput contents], impl_->gemvShape[0] * sizeof(float));
}

bool GemvMpsKernel::in_progress() const noexcept {
    if (impl_->lastCommandBuffer == nil) {
        return false;
    }
    const auto status = impl_->lastCommandBuffer.status;
    return status != MTLCommandBufferStatusCompleted &&
           status != MTLCommandBufferStatusError;
}

}
