#pragma once

#include <memory>
#include <string>
#include <limits>
#include <stdexcept>
#include <type_traits>

#include "llmetal/tensor.hpp"

namespace llmetal {

// Tensor elements should be stored by value
template<typename T>
concept TensorElement = std::is_trivially_copyable_v<T> && !std::is_const_v<T>;

class MetalContext {
public:
    MetalContext();
    ~MetalContext();

    MetalContext(const MetalContext&) = delete;
    MetalContext& operator=(const MetalContext&) = delete;

    MetalContext(MetalContext&&) = delete;
    MetalContext& operator=(MetalContext&&) = delete;
    
    [[nodiscard]] std::string device_name() const;

    [[nodiscard]] void* device_handle() const;        // MTLDevice
    [[nodiscard]] void* command_queue_handle() const; // MTLCommandQueue

    template<TensorElement T>
    [[nodiscard]] GpuTensor<T> allocate(Shape shape) const;
    template<TensorElement T>
    [[nodiscard]] GpuTensor<T> upload(const CpuTensor<T>& source) const;
    template<TensorElement T>
    [[nodiscard]] CpuTensor<T> download(const GpuTensor<T>& source) const;

private:
    class Impl;
    std::unique_ptr<Impl> impl_;

};

template<TensorElement T>
[[nodiscard]] GpuTensor<T> MetalContext::allocate(Shape shape) const {
    const std::size_t count = shape.numel();
    
    if (count > std::numeric_limits<std::size_t>::max() / sizeof(T)) {
        throw std::runtime_error("GPU Tensor bytes exceeds maximum");
    }

    std::shared_ptr<detail::GpuStorage> storage = detail::make_gpu_storage(
        device_handle(), count * sizeof(T)
    );

    return GpuTensor<T>(std::move(shape), std::move(storage));
}

template<TensorElement T>
[[nodiscard]] GpuTensor<T> MetalContext::upload(const CpuTensor<T>& source) const {
    GpuTensor<T> destination = allocate<T>(source.shape());
    detail::copy_to_storage(destination.storage_, 
                            source.data(), 
                            source.byte_size(), 
                            destination.byte_offset_);
    return destination;
}

template<TensorElement T>
[[nodiscard]] CpuTensor<T> MetalContext::download(const GpuTensor<T>& source) const {
    CpuTensor<T> destination(source.shape());
    detail::copy_from_storage(destination.data(), 
                              source.storage_, 
                              source.byte_size(), 
                              source.byte_offset_);
    return destination;
}

} // namespace llmetal
