#pragma once

#include <memory>
#include <string>

namespace llmetal {

class MetalContext {
public:
    MetalContext();
    ~MetalContext();

    MetalContext(const MetalContext&) = delete;
    MetalContext& operator=(const MetalContext&) = delete;

    MetalContext(MetalContext&&) noexcept;
    MetalContext& operator=(MetalContext&&) noexcept;
    
    [[nodiscard]] std::string device_name() const;

    [[nodiscard]] void* device_handle() const;        // MTLDevice
    [[nodiscard]] void* command_queue_handle() const; // MTLCommandQueue

private:
    class Impl;
    std::unique_ptr<Impl> impl_;

};

} // namespace llmetal
