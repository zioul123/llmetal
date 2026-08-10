#include <fstream>
#include <iostream>
#include <vector>

template <typename T>
std::vector<T> read_raw(
    const std::filesystem::path& path,
    std::size_t expected_elements
) {
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file) {
        throw std::runtime_error("Could not open: " + path.string());
    }

    const std::streamsize byte_count = file.tellg();
    const std::size_t expected_bytes = expected_elements * sizeof(T);

    if (byte_count < 0 ||
        static_cast<std::size_t>(byte_count) != expected_bytes) {
        throw std::runtime_error(
            "Unexpected file size for " + path.string() +
            ": expected " + std::to_string(expected_bytes) +
            " bytes, got " + std::to_string(byte_count)
        );
    }

    file.seekg(0, std::ios::beg);

    std::vector<T> values(expected_elements);
    if (!file.read(
            reinterpret_cast<char*>(values.data()),
            static_cast<std::streamsize>(expected_bytes)
        )) {
        throw std::runtime_error("Could not read: " + path.string());
    }

    return values;
}
