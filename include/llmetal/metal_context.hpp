#pragma once

#include <memory>
#include <string>
#include <limits>
#include <stdexcept>

#include "llmetal/tensor.hpp"

namespace llmetal {

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

    template<typename T>
    [[nodiscard]] GpuTensor<T> allocate(Shape shape) const;
    template<typename T>
    [[nodiscard]] GpuTensor<T> upload(const CpuTensor<T>& source) const;
    template<typename T>
    [[nodiscard]] CpuTensor<T> download(const GpuTensor<T>& source) const;

private:
    class Impl;
    std::unique_ptr<Impl> impl_;

};

} // namespace llmetal
