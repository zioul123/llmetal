cmake_minimum_required(VERSION 3.25...4.0)

project(LLMetal LANGUAGES CXX OBJCXX)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

add_executable(LLMetal)

target_sources(LLMetal
  PRIVATE
    src/main.cpp
    src/metal_context.mm
    src/vector_add.mm
)

target_include_directories(LLMetal
  PRIVATE
    include
)

# target_compile_features(LLMetal PRIVATE cxx_std_20)
set_target_properties(LLMetal PROPERTIES 
  CXX_STANDARD 20
  CXX_STANDARD_REQUIRED ON
  CXX_EXTENSIONS OFF

  OBJCXX_STANDARD 20
  OBJCXX_STANDARD_REQUIRED ON
  OBJCXX_EXTENSIONS OFF
)

target_compile_definitions(LLMetal
  PRIVATE 
    LLMETAL_BUILD_TYPE="$<CONFIG>"
)

target_compile_options(LLMetal PRIVATE
  "$<$<COMPILE_LANGUAGE:OBJCXX>:-fobjc-arc>"
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

# === Metal shader compilation ===
find_program(XCRUN xcrun REQUIRED)

set(METAL_SOURCE ${CMAKE_CURRENT_SOURCE_DIR}/shaders/vector_add.metal)
set(METAL_AIR ${CMAKE_CURRENT_BINARY_DIR}/shaders/vector_add.air)
set(METAL_LIB ${CMAKE_CURRENT_BINARY_DIR}/shaders/vector_add.metallib)

add_custom_command(
  OUTPUT ${METAL_AIR}
  COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_CURRENT_BINARY_DIR}/shaders
  COMMAND ${XCRUN} metal -c ${METAL_SOURCE} -o ${METAL_AIR}
  DEPENDS ${METAL_SOURCE}
  VERBATIM
)

add_custom_command(
  OUTPUT ${METAL_LIB}
  COMMAND ${XCRUN} metallib ${METAL_AIR} -o ${METAL_LIB}
  DEPENDS ${METAL_AIR}
  VERBATIM
)

add_custom_target(MetalLibs ALL
  DEPENDS ${METAL_LIB}
)

add_dependencies(LLMetal MetalLibs)

# === Tests ===
include(CTest)

if(BUILD_TESTING)
  add_executable(LLMetalVectorAddSmoke)

  target_sources(LLMetalVectorAddSmoke
    PRIVATE
      tests/vector_add_smoke.cpp
      src/metal_context.mm
      src/vector_add.mm
  )

  target_include_directories(LLMetalVectorAddSmoke
    PRIVATE
      include
  )

  set_target_properties(LLMetalVectorAddSmoke PROPERTIES
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED ON
    CXX_EXTENSIONS OFF

    OBJCXX_STANDARD 20
    OBJCXX_STANDARD_REQUIRED ON
    OBJCXX_EXTENSIONS OFF
  )

  target_compile_options(LLMetalVectorAddSmoke PRIVATE
    "$<$<COMPILE_LANGUAGE:OBJCXX>:-fobjc-arc>"
  )

  target_link_libraries(LLMetalVectorAddSmoke
    PRIVATE
      "-framework Metal"
      "-framework Foundation"
  )

  add_dependencies(LLMetalVectorAddSmoke MetalLibs)

  add_test(NAME vector_add_smoke COMMAND LLMetalVectorAddSmoke)
endif()
