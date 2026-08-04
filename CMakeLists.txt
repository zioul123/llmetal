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
    "-framework MetalPerformanceShaders"
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

set(LLMETAL_SHADER_NAMES
  vector_add
  gemv
)

set(LLMETAL_METALLIBS)

foreach(shader_name IN LISTS LLMETAL_SHADER_NAMES)
  set(shader_source ${CMAKE_CURRENT_SOURCE_DIR}/shaders/${shader_name}.metal)
  set(shader_air ${CMAKE_CURRENT_BINARY_DIR}/shaders/${shader_name}.air)
  set(shader_metallib ${CMAKE_CURRENT_BINARY_DIR}/shaders/${shader_name}.metallib)

  add_custom_command(
    OUTPUT ${shader_air}
    COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_CURRENT_BINARY_DIR}/shaders
    COMMAND ${XCRUN} metal -c ${shader_source} -o ${shader_air}
    DEPENDS ${shader_source}
    VERBATIM
  )

  add_custom_command(
    OUTPUT ${shader_metallib}
    COMMAND ${XCRUN} metallib ${shader_air} -o ${shader_metallib}
    DEPENDS ${shader_air}
    VERBATIM
  )

  list(APPEND LLMETAL_METALLIBS ${shader_metallib})
endforeach()

add_custom_target(MetalLibs ALL
  DEPENDS ${LLMETAL_METALLIBS}
)

add_dependencies(LLMetal MetalLibs)

# === Benchmarks ===
option(LLMETAL_BUILD_BENCHMARKS "Build LLMetal benchmarks" ON)

if(LLMETAL_BUILD_BENCHMARKS)
  add_executable(LLMetalGemvBenchmark
    benchmarks/gemv_benchmark.mm
    src/metal_context.mm
    src/gemv_naive.mm
    src/gemv_mps.mm
    src/gemv_interleaved.mm
    src/cpu/gemv.cpp
    src/metal_job.mm
  )

  target_include_directories(LLMetalGemvBenchmark PRIVATE include)

  set_target_properties(LLMetalGemvBenchmark PROPERTIES
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED ON
    CXX_EXTENSIONS OFF
    OBJCXX_STANDARD 20
    OBJCXX_STANDARD_REQUIRED ON
    OBJCXX_EXTENSIONS OFF
  )

  target_compile_options(LLMetalGemvBenchmark PRIVATE
    "$<$<COMPILE_LANGUAGE:OBJCXX>:-fobjc-arc>"
  )

  target_link_libraries(LLMetalGemvBenchmark
    PRIVATE
      "-framework Metal"
      "-framework Foundation"
      "-framework MetalPerformanceShaders"
  )

  add_dependencies(LLMetalGemvBenchmark MetalLibs)
endif()


# === Tests ===
include(CTest)

if(BUILD_TESTING)
  # --- LLMetalVectorAddSmoke ---

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
      "-framework MetalPerformanceShaders"
  )

  add_dependencies(LLMetalVectorAddSmoke MetalLibs)

  add_test(NAME vector_add_smoke COMMAND LLMetalVectorAddSmoke)

  # --- LLMetalGemvSmoke ---

  add_executable(LLMetalGemvSmoke)

  target_sources(LLMetalGemvSmoke
    PRIVATE
      tests/gemv_smoke.cpp
      src/metal_context.mm
      src/gemv_naive.mm
      src/gemv_mps.mm
      src/gemv_interleaved.mm
      src/metal_job.mm
  )

  target_include_directories(LLMetalGemvSmoke
    PRIVATE
      include
  )

  set_target_properties(LLMetalGemvSmoke PROPERTIES
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED ON
    CXX_EXTENSIONS OFF

    OBJCXX_STANDARD 20
    OBJCXX_STANDARD_REQUIRED ON
    OBJCXX_EXTENSIONS OFF
  )

  target_compile_options(LLMetalGemvSmoke PRIVATE
    "$<$<COMPILE_LANGUAGE:OBJCXX>:-fobjc-arc>"
  )

  target_link_libraries(LLMetalGemvSmoke
    PRIVATE
      "-framework Metal"
      "-framework Foundation"
      "-framework MetalPerformanceShaders"
  )

  add_dependencies(LLMetalGemvSmoke MetalLibs)

  add_test(NAME gemv_smoke COMMAND LLMetalGemvSmoke)

  # --- LLMetalGemvInterleaveSmoke ---

  add_executable(LLMetalGemvInterleaveSmoke)

  target_sources(LLMetalGemvInterleaveSmoke
    PRIVATE
      tests/gemv_interleave_smoke.cpp
      src/metal_context.mm
      src/gemv_interleaved.mm
      src/metal_job.mm
  )

  target_include_directories(LLMetalGemvInterleaveSmoke
    PRIVATE
      include
  )

  set_target_properties(LLMetalGemvInterleaveSmoke PROPERTIES
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED ON
    CXX_EXTENSIONS OFF

    OBJCXX_STANDARD 20
    OBJCXX_STANDARD_REQUIRED ON
    OBJCXX_EXTENSIONS OFF
  )

  target_compile_options(LLMetalGemvInterleaveSmoke PRIVATE
    "$<$<COMPILE_LANGUAGE:OBJCXX>:-fobjc-arc>"
  )

  target_link_libraries(LLMetalGemvInterleaveSmoke
    PRIVATE
      "-framework Metal"
      "-framework Foundation"
      "-framework MetalPerformanceShaders"
  )

  add_dependencies(LLMetalGemvInterleaveSmoke MetalLibs)

  add_test(NAME gemv_interleave_smoke COMMAND LLMetalGemvInterleaveSmoke)
endif()
