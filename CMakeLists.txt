cmake_minimum_required(VERSION 3.22)
project(cuda_inference_kernels LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

if(MSVC AND NOT CMAKE_CUDA_RUNTIME_LIBRARY)
  set(CMAKE_CUDA_RUNTIME_LIBRARY Shared)
endif()

if(NOT CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 70;80;89)
endif()

function(configure_cuda_target target)
  target_include_directories(${target} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
  target_compile_options(${target} PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr;-lineinfo>
  )
endfunction()

add_executable(rmsnorm_bench src/rmsnorm_bench.cu src/rmsnorm_lib.cu)
configure_cuda_target(rmsnorm_bench)

add_executable(rmsnorm_tests tests/test_rmsnorm.cu src/rmsnorm_lib.cu)
configure_cuda_target(rmsnorm_tests)

enable_testing()
add_test(NAME rmsnorm_tests COMMAND rmsnorm_tests)
