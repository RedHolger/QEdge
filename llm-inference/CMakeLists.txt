cmake_minimum_required(VERSION 3.23)
project(llm-inference LANGUAGES CXX C)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
include(GNUInstallDirs)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Build type" FORCE)
endif()

if(APPLE)
  set(CMAKE_OSX_DEPLOYMENT_TARGET "14.0" CACHE STRING "macOS deployment target")
endif()

option(ENABLE_QNN_STUB "Build QnnStub backend and power-efficiency benchmark" ON)
option(ENABLE_SANITIZERS "Build with AddressSanitizer and UndefinedBehaviorSanitizer" OFF)
option(ENABLE_METAL "Build the experimental Metal backend" ON)

if(NOT APPLE)
  message(FATAL_ERROR
    "The current C++ kernel lab requires macOS because it links Accelerate and "
    "Metal. The Python packages remain portable. See cpp/README.md for the "
    "kernel-lab boundary.")
endif()

if(NOT ENABLE_METAL)
  message(FATAL_ERROR "llm_infer currently requires the Metal target; leave ENABLE_METAL=ON for Stage 01.")
endif()

find_library(ACCELERATE_FW Accelerate REQUIRED)
find_library(FOUNDATION_FW Foundation REQUIRED)
find_library(METAL_FW Metal REQUIRED)
find_library(METAL_PERFORMANCE_SHADERS_FW MetalPerformanceShaders REQUIRED)

set(QEDGE_KERNEL_ARCH_FLAGS "")
if(CMAKE_SYSTEM_PROCESSOR MATCHES "arm64|aarch64")
  set(QEDGE_KERNEL_ARCH_FLAGS -march=armv8.5-a+fp16+rcpc+dotprod)
endif()

function(qedge_apply_flags target)
  if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    target_compile_options(${target} PRIVATE -O0 -g)
  else()
    target_compile_options(${target} PRIVATE -O3 ${QEDGE_KERNEL_ARCH_FLAGS} -ffast-math)
  endif()
  if(ENABLE_SANITIZERS)
    target_compile_options(${target} PRIVATE -fsanitize=address,undefined -fno-omit-frame-pointer)
    target_link_options(${target} PRIVATE -fsanitize=address,undefined)
  endif()
endfunction()

set(CORE_SOURCES
  src/core/tensor.cpp
  src/core/config.cpp
  src/kv_cache/kv_cache.cpp
  src/kv_cache/policy.cpp
  src/backend/backend.cpp
  src/backend/cpu_backend.cpp
  src/benchmark/benchmark.cpp
)

if(ENABLE_QNN_STUB)
  list(APPEND CORE_SOURCES
    src/backend/qnn_stub_backend.cpp
    src/benchmark/power_report.cpp
  )
endif()

# Core library (CPU + KV cache + benchmark + optional QNN stub)
add_library(llm_core ${CORE_SOURCES})
target_include_directories(llm_core PUBLIC include)
target_link_libraries(llm_core PRIVATE ${ACCELERATE_FW})
qedge_apply_flags(llm_core)
if(ENABLE_QNN_STUB)
  target_compile_definitions(llm_core PUBLIC ENABLE_QNN_STUB=1)
endif()

# Metal backend (experimental kernel-lab path, not a production runtime)
add_library(llm_metal src/backend/metal_backend.mm)
target_include_directories(llm_metal PUBLIC include metal)
target_link_libraries(llm_metal PRIVATE ${FOUNDATION_FW} ${METAL_FW} ${METAL_PERFORMANCE_SHADERS_FW})
target_compile_features(llm_metal PRIVATE cxx_std_20)
qedge_apply_flags(llm_metal)

# Main synthetic kernel-lab binary
add_executable(llm_infer src/main.cpp)
target_link_libraries(llm_infer PRIVATE llm_core llm_metal)
qedge_apply_flags(llm_infer)

# Combined KV-cache test binary (reports its own sub-test totals).
add_executable(run_tests tests/test_kv_cache.cpp)
target_link_libraries(run_tests PRIVATE llm_core)
qedge_apply_flags(run_tests)

# Independent scalar-reference attention and RoPE safety regressions.
add_executable(run_kernel_tests tests/test_attention.cpp)
target_link_libraries(run_kernel_tests PRIVATE llm_core)
qedge_apply_flags(run_kernel_tests)

# CTest: the legacy test binaries report their own sub-test totals.
enable_testing()
add_test(NAME llm_kv_cache COMMAND run_tests)
set_tests_properties(llm_kv_cache PROPERTIES LABELS "cpu;stage01")
add_test(NAME kernel_safety COMMAND run_kernel_tests)
set_tests_properties(kernel_safety PROPERTIES LABELS "cpu;kernel;stage01")

if(ENABLE_QNN_STUB)
  add_executable(llm_qnn_bench src/main_qnn.cpp)
  target_link_libraries(llm_qnn_bench PRIVATE llm_core llm_metal)
  qedge_apply_flags(llm_qnn_bench)

  add_executable(run_qnn_tests tests/test_qnn_stub.cpp)
  target_link_libraries(run_qnn_tests PRIVATE llm_core)
  qedge_apply_flags(run_qnn_tests)

  add_test(NAME qnn_stub_cpu COMMAND run_qnn_tests)
  set_tests_properties(qnn_stub_cpu PROPERTIES LABELS "cpu;qnn-stub;stage01")

  message(STATUS "QNN Stub enabled — CPU-only stub, no NPU measurement")
endif()

install(TARGETS llm_core llm_metal llm_infer
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR})
if(ENABLE_QNN_STUB)
  install(TARGETS llm_qnn_bench
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
endif()
install(DIRECTORY include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "Sanitizers: ${ENABLE_SANITIZERS}")
message(STATUS "CTest: llm_kv_cache, kernel_safety and optional qnn_stub_cpu")
