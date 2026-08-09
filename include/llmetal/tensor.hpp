#pragma once

#include <initializer_list>
#include <type_traits>
#include <vector>
#include <iostream>
#include <sstream>
#include <iomanip>
#include <span>

#include "llmetal/overflow_helpers.hpp"

namespace llmetal {

class Shape {
public:
    Shape(std::initializer_list<std::size_t> dims): dims_(dims) {}
    explicit Shape(std::vector<std::size_t> dims): dims_(std::move(dims)) {}
    
    std::size_t rank() const noexcept { return dims_.size(); }
    std::size_t numel() const { 
        std::size_t result = 1;
        for (const std::size_t dimension : dims_) {
            result = checked_multiply(result, dimension);
        }
        return result;
    }
    std::span<const std::size_t> dims() const noexcept { return dims_; }
    
    std::size_t operator[](std::size_t axis) const { return dims_.at(axis); }
    friend std::ostream& operator<<(std::ostream& os, const Shape& shape);
    std::string to_string() const {
        std::ostringstream oss; oss << *this; return oss.str();
    }

private:
    std::vector<std::size_t> dims_;
};
inline std::ostream& operator<<(std::ostream& os, const Shape& shape) {
    os << "(";
    for (std::size_t d = 0; d < shape.rank(); d++)
        os << (d == 0 ? "" : " ") << shape.dims()[d];
    os << ")";
    return os;
}

template <typename T>
class CpuTensor {
public:
    explicit CpuTensor(Shape shape): shape_(std::move(shape)), data_(shape_.numel()) {}
    CpuTensor(Shape shape, std::vector<T> data)
        : shape_(std::move(shape)), data_(std::move(data)) {
        if (data_.size() != shape_.numel()) throw std::runtime_error("Data size does not match shape size");
    }
    
    const Shape& shape() const noexcept { return shape_; }
    std::size_t numel() const { return data_.size(); }
    std::size_t byte_size() const { return checked_multiply(data_.size(), sizeof(T)); }
    T* data() noexcept { return data_.data(); }
    const T* data() const noexcept { return data_.data(); }
    std::span<T> span() noexcept { return data_; }
    std::span<const T> span() const noexcept { return data_; }

    // Allow reading and assignment to data directly
    T operator[](std::size_t index) const { return data_[index]; }
    T& operator[](std::size_t index) { return data_[index]; }

private:
    Shape shape_;
    std::vector<T> data_;
};
template <typename T>
inline std::ostream& operator<<(std::ostream& os, const CpuTensor<T>& tensor) {
    const Shape& shape = tensor.shape();

    // Iterate through each index.
    std::vector<std::size_t> indices(shape.rank(), 0);
    
    for (std::size_t i = 0; i < tensor.numel(); ++i) {
        os << std::setw(10) << tensor[i] << " ";
        // Increment last dimension
        ++indices.back();

        // Propagate carry-overs from right to left
        for (std::size_t d = shape.rank() - 1; d > 0; --d) {
            // No carry over - done.
            if (indices[d] < shape[d]) break;
            
            // Carry over and print newline
            indices[d] = 0;
            if (d > 0) ++indices[d - 1];
            os << '\n';
        }
    }
    return os;
}

namespace detail {

class GpuStorage;

std::shared_ptr<detail::GpuStorage> make_gpu_storage(
    void* device_handle,
    std::size_t byte_size
);

void copy_to_storage(const std::shared_ptr<GpuStorage>& storage,
                     const void* src,
                     std::size_t byte_size,
                     std::size_t offset);

void copy_from_storage(void* dst,
                       const std::shared_ptr<GpuStorage>& storage,
                       std::size_t byte_size,
                       std::size_t offset);

void* native_buffer(const std::shared_ptr<GpuStorage>& storage) noexcept;

} // namespace detail

template <typename T>
class GpuTensor {
    static_assert(std::is_trivially_copyable_v<T>, "T must be trivially copyable");

public:
    GpuTensor(const GpuTensor&) noexcept = default;
    GpuTensor& operator=(const GpuTensor&) noexcept = default;
    GpuTensor(GpuTensor&&) noexcept = default;
    GpuTensor& operator=(GpuTensor&&) noexcept = default;
    ~GpuTensor() = default;
    
    [[nodiscard]] const Shape& shape() const noexcept { return shape_; }
    [[nodiscard]] std::size_t numel() const { return shape_.numel(); }
    [[nodiscard]] std::size_t byte_size() const { return checked_multiply(shape_.numel(), sizeof(T)); }

private:
    GpuTensor(Shape shape, std::shared_ptr<detail::GpuStorage> storage, std::size_t byte_offset = 0)
        : shape_(std::move(shape)), storage_(std::move(storage)), byte_offset_(byte_offset) {}
    [[nodiscard]] void* buffer_handle() const noexcept { return detail::native_buffer(storage_); }
    
    Shape shape_;
    std::shared_ptr<detail::GpuStorage> storage_;
    std::size_t byte_offset_ = 0;

    friend class MetalContext;
    friend class EmbeddingKernel;
    friend class GemvNaiveKernel;
    friend class GemvNRPSGKernel;
};

}
