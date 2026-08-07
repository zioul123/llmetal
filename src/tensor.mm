#include <llmetal/tensor.hpp>

#include <cstring>

#include <Foundation/Foundation.h>
#include <Metal/Metal.h>

namespace llmetal::detail {

class GpuStorage {
public:
    GpuStorage(id<MTLBuffer> buffer, std::size_t byte_capacity)
        : buffer_(buffer), capacity_(byte_capacity) {}
    id<MTLBuffer> buffer_ = nil;
    std::size_t capacity_ = 0;
};

std::shared_ptr<detail::GpuStorage> make_gpu_storage(
    void* device_handle,
    std::size_t byte_size
) {
    if (byte_size == 0) throw std::runtime_error("Cannot create a buffer with zero size");

    id<MTLDevice> device = (__bridge id<MTLDevice>)device_handle;
    id<MTLBuffer> buffer = [device newBufferWithLength:byte_size options:MTLResourceStorageModeShared];
    if (buffer == nil) throw std::runtime_error("Failed to create buffer");

    return std::make_shared<GpuStorage>(buffer, byte_size);
}

void copy_to_storage(const std::shared_ptr<GpuStorage>& storage,
                     const void* src,
                     std::size_t byte_size,
                     std::size_t offset) {
    if (!storage || offset > storage->capacity_ ||
        byte_size > storage->capacity_ - offset) {
        throw std::out_of_range("GPU Upload exceeds buffer bounds");
    }

    void* dst = static_cast<std::byte*>([storage->buffer_ contents]) + offset;
    std::memcpy(dst, src, byte_size);
}


void copy_from_storage(void* dst,
                       const std::shared_ptr<GpuStorage>& storage,
                       std::size_t byte_size,
                       std::size_t offset) {
    if (!storage || offset > storage->capacity_ ||
        byte_size > storage->capacity_ - offset) {
        throw std::out_of_range("GPU download exceeds buffer bounds");
    }

    void* src = static_cast<std::byte*>([storage->buffer_ contents]) + offset;
    std::memcpy(dst, src, byte_size);
}

void* native_buffer(const std::shared_ptr<GpuStorage>& storage) noexcept {
    return storage ? (__bridge void*)storage->buffer_ : nullptr;
}

} // namespace llmetal::detail

// template <typename T>
// class GpuTensor<T>::GpuStorage {
// public:
//     explicit GpuStorage(Shape shape): shape_(std::move(shape)) {}
//     const Shape& shape() const noexcept { return shape_; }
//     std::size_t numel() const { return shape_.numel(); }
// private:
//     Shape shape_;
// };

// template <typename T>
// GpuTensor<T>::GpuStorage(Shape shape): {
// }


// } // namespace llmetal
