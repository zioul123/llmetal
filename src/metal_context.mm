#include <llmetal/metal_context.hpp>

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <iostream>
#include <stdexcept>

void print_metal_device() {
    @autoreleasepool {
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();

        if (device == nil) {
            throw std::runtime_error("Metal is unavailable: no default Metal device was found");
        }

        const char* device_name = device.name.UTF8String;

        std::cout << "Metal device: " << (device_name != nullptr ? device_name : "<unknown>") << std::endl;
    }
}