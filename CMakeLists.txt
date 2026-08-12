cmake_minimum_required(VERSION 3.25...4.0)

project(LLMetal LANGUAGES CXX OBJCXX)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

add_executable(LLMetal)

target_sources(LLMetal
  PRIVATE
    src/main.cpp
    src/metal_context.mm
    src/tensor.mm
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
  embedding
  rms_norm
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
  # --- Gemv benchmark ---
  add_executable(LLMetalGemvBenchmark
    benchmarks/gemv_benchmark.mm
    src/metal_context.mm
    src/tensor.mm
    src/gemv_naive.mm
    src/gemv_mps.mm
    src/gemv_NRPSG.mm
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

  # --- Embedding benchmark ---
  add_executable(LLMetalEmbeddingBenchmark
    benchmarks/embedding_benchmark.mm
    src/metal_context.mm
    src/metal_job.mm
    src/tensor.mm
    src/embedding.mm
    src/cpu/embedding.cpp
  )

  target_include_directories(LLMetalEmbeddingBenchmark PRIVATE include)

  set_target_properties(LLMetalEmbeddingBenchmark PROPERTIES
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED ON
    CXX_EXTENSIONS OFF
    OBJCXX_STANDARD 20
    OBJCXX_STANDARD_REQUIRED ON
    OBJCXX_EXTENSIONS OFF
  )

  target_compile_options(LLMetalEmbeddingBenchmark PRIVATE
    "$<$<COMPILE_LANGUAGE:OBJCXX>:-fobjc-arc>"
  )

  target_link_libraries(LLMetalEmbeddingBenchmark
    PRIVATE
      "-framework Metal"
      "-framework Foundation"
      "-framework MetalPerformanceShaders"
  )

  add_dependencies(LLMetalEmbeddingBenchmark MetalLibs)

  # --- RmsNorm benchmark ---
  add_executable(LLMetalRmsNormBenchmark
    benchmarks/rms_norm_benchmark.mm
    src/metal_context.mm
    src/metal_job.mm
    src/tensor.mm
    src/rms_norm.mm
    src/cpu/rms_norm.cpp
  )

  target_include_directories(LLMetalRmsNormBenchmark PRIVATE include)

  set_target_properties(LLMetalRmsNormBenchmark PROPERTIES
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED ON
    CXX_EXTENSIONS OFF
    OBJCXX_STANDARD 20
    OBJCXX_STANDARD_REQUIRED ON
    OBJCXX_EXTENSIONS OFF
  )

  target_compile_options(LLMetalRmsNormBenchmark PRIVATE
    "$<$<COMPILE_LANGUAGE:OBJCXX>:-fobjc-arc>"
  )

  target_link_libraries(LLMetalRmsNormBenchmark
    PRIVATE
      "-framework Metal"
      "-framework Foundation"
      "-framework MetalPerformanceShaders"
  )

  add_dependencies(LLMetalRmsNormBenchmark MetalLibs)
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
      src/tensor.mm
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
      tests/gemv_smoke.mm
      src/metal_context.mm
      src/tensor.mm
      src/gemv_naive.mm
      src/gemv_mps.mm
      src/gemv_NRPSG.mm
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

  # --- LLMetalEmbeddingSmoke ---

  add_executable(LLMetalEmbeddingSmoke)

  target_sources(LLMetalEmbeddingSmoke
    PRIVATE
      tests/embedding_smoke.cpp
      src/cpu/embedding.cpp
      src/metal_context.mm
      src/embedding.mm
      src/tensor.mm
      src/metal_job.mm
  )

  target_include_directories(LLMetalEmbeddingSmoke
    PRIVATE
      include
  )

  set_target_properties(LLMetalEmbeddingSmoke PROPERTIES
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED ON
    CXX_EXTENSIONS OFF

    OBJCXX_STANDARD 20
    OBJCXX_STANDARD_REQUIRED ON
    OBJCXX_EXTENSIONS OFF
  )

  target_compile_options(LLMetalEmbeddingSmoke PRIVATE
    "$<$<COMPILE_LANGUAGE:OBJCXX>:-fobjc-arc>"
  )

  target_link_libraries(LLMetalEmbeddingSmoke
    PRIVATE
      "-framework Metal"
      "-framework Foundation"
      "-framework MetalPerformanceShaders"
  )

  add_dependencies(LLMetalEmbeddingSmoke MetalLibs)

  add_test(NAME embedding COMMAND LLMetalEmbeddingSmoke)
  set_tests_properties(embedding PROPERTIES
    WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
  )

  # --- LLMetalRmsNormSmoke ---

  add_executable(LLMetalRmsNormSmoke)

  target_sources(LLMetalRmsNormSmoke
    PRIVATE
      tests/rms_norm_smoke.cpp
      src/cpu/rms_norm.cpp
      src/rms_norm.mm
      src/metal_context.mm
      src/tensor.mm
      src/metal_job.mm
  )

  target_include_directories(LLMetalRmsNormSmoke
    PRIVATE
      include
  )

  set_target_properties(LLMetalRmsNormSmoke PROPERTIES
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED ON
    CXX_EXTENSIONS OFF

    OBJCXX_STANDARD 20
    OBJCXX_STANDARD_REQUIRED ON
    OBJCXX_EXTENSIONS OFF
  )

  target_compile_options(LLMetalRmsNormSmoke PRIVATE
    "$<$<COMPILE_LANGUAGE:OBJCXX>:-fobjc-arc>"
  )

  target_link_libraries(LLMetalRmsNormSmoke
    PRIVATE
      "-framework Metal"
      "-framework Foundation"
      "-framework MetalPerformanceShaders"
  )

  add_dependencies(LLMetalRmsNormSmoke MetalLibs)

  add_test(NAME rms_norm COMMAND LLMetalRmsNormSmoke)
  set_tests_properties(rms_norm PROPERTIES
    WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
  )

  # --- LLMetalRoPESmoke ---

  add_executable(LLMetalRoPESmoke)

  target_sources(LLMetalRoPESmoke
    PRIVATE
      tests/rms_norm_smoke.cpp
      src/cpu/rms_norm.cpp
      src/rms_norm.mm
      src/metal_context.mm
      src/tensor.mm
      src/metal_job.mm
  )

  target_include_directories(LLMetalRoPESmoke
    PRIVATE
      include
  )

  set_target_properties(LLMetalRoPESmoke PROPERTIES
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED ON
    CXX_EXTENSIONS OFF

    OBJCXX_STANDARD 20
    OBJCXX_STANDARD_REQUIRED ON
    OBJCXX_EXTENSIONS OFF
  )

  target_compile_options(LLMetalRoPESmoke PRIVATE
    "$<$<COMPILE_LANGUAGE:OBJCXX>:-fobjc-arc>"
  )

  target_link_libraries(LLMetalRoPESmoke
    PRIVATE
      "-framework Metal"
      "-framework Foundation"
      "-framework MetalPerformanceShaders"
  )

  add_dependencies(LLMetalRoPESmoke MetalLibs)

  add_test(NAME rope COMMAND LLMetalRoPESmoke)
  set_tests_properties(rope PROPERTIES
    WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
  )
endif()
