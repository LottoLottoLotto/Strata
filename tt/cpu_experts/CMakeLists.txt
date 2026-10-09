# libfnx_cpu.so: Strata's CPU expert kernels behind a C ABI, for the Tenstorrent port (fnx_cpu.cpp).
# Build it with tt/cpu_experts/build.sh (in the TT image).  By hand:
#   cmake -S tt/cpu_experts -B tt/cpu_experts/build-tt -G Ninja [-DSTRATA_GGML_DIR=<llama.cpp source>]
# Builds this repository's strata_kernels_cpu + ggml-cpu (no CUDA) as position-independent code and links them in.
# STRATA_SRC defaults to the repository that contains tt/; STRATA_GGML_DIR is Strata's own option (empty: Strata
# fetches its pinned llama.cpp commit, third_party/ggml/VERSION.txt).
cmake_minimum_required(VERSION 3.24)
project(fnx_cpu LANGUAGES C CXX)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
get_filename_component(_fnx_repo "${CMAKE_CURRENT_SOURCE_DIR}/../.." ABSOLUTE)
set(STRATA_SRC "${_fnx_repo}" CACHE PATH "Strata checkout (default: the repository containing tt/)")
if(NOT EXISTS "${STRATA_SRC}/CMakeLists.txt")
  message(FATAL_ERROR "STRATA_SRC must point at a Strata checkout")
endif()
set(STRATA_ENABLE_CUDA OFF CACHE BOOL "" FORCE)
set(STRATA_ENABLE_HIP OFF CACHE BOOL "" FORCE)
set(STRATA_BUILD_TESTS OFF CACHE BOOL "" FORCE)
add_subdirectory(${STRATA_SRC} strata EXCLUDE_FROM_ALL)
add_library(fnx_cpu SHARED fnx_cpu.cpp)
target_link_libraries(fnx_cpu PRIVATE strata_kernels_cpu)
target_include_directories(fnx_cpu PRIVATE ${STRATA_SRC}/include)
