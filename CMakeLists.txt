cmake_minimum_required(VERSION 3.25...4.0)

project(LLMetal LANGUAGES CXX OBJCXX)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

add_executable(LLMetal)

target_sources(LLMetal
  PRIVATE
    src/main.cpp
    src/metal_context.mm
)

target_include_directories(LLMetal
  PRIVATE
    include
)

target_compile_features(LLMetal PRIVATE cxx_std_20)

target_compile_definitions(LLMetal
  PRIVATE 
    LLMETAL_BUILD_TYPE="$<CONFIG>"
)

target_link_libraries(LLMetal
  PRIVATE
    "-framework Metal"
    "-framework Foundation"
)

# Explicit version that will throw error instead
# find_library(METAL_FRAMEWORK Metal REQUIRED)
# find_library(FOUNDATION_FRAMEWORK Foundation REQUIRED)
# target_link_libraries(LLMetal
#   PRIVATE 
#     ${METAL_FRAMEWORK}
#     ${FOUNDATION_FRAMEWORK}
# )
