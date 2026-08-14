#include <cstddef>
#include <limits>
#include <stdexcept>
#include <string_view>
#include <string>

// Check that the multiplication of two size_t values does not overflow.
[[nodiscard]] constexpr std::size_t checked_multiply(
    std::size_t lhs,
    std::size_t rhs
) {
    if (rhs != 0 &&
        lhs > std::numeric_limits<std::size_t>::max() / rhs) {
        throw std::overflow_error("Shape element count overflow");
    }

    return lhs * rhs;
}

[[nodiscard]] constexpr std::uint32_t checked_multiply_u32(
    std::uint32_t lhs,
    std::uint32_t rhs
) {
    if (rhs != 0 &&
        lhs > std::numeric_limits<std::uint32_t>::max() / rhs) {
        throw std::overflow_error("Shape element count overflow");
    }

    return lhs * rhs;
}

[[nodiscard]] inline std::uint32_t checked_u32(
    std::size_t value,
    std::string_view name
) {
    if (value > std::numeric_limits<std::uint32_t>::max()) {
        throw std::overflow_error(
            std::string(name) + " exceeds Metal uint range"
        );
    }

    return static_cast<std::uint32_t>(value);
}
