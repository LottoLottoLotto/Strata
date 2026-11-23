# libfnx_cpu.so: Strata's CPU expert kernels behind a C ABI, for the Tenstorrent port (fnx_cpu.cpp).
# Build it with tt/cpu_experts/build.sh (in the TT image).  By hand:
#   cmake -S tt/cpu_experts -B tt/cpu_experts/build-tt -G Ninja [-DSTRATA_GGML_DIR=<llama.cpp source>]
# Builds this repository's strata_kernels_cpu + ggml-cpu (no CUDA) as position-independent code and links them in.
# STRATA_SRC defaults to the repository that contains tt/; STRATA_GGML_DIR is Strata's own option (empty: Strata
# fetches its pinned llama.cpp commit, third_party/ggml/VERSION.txt).
# As a Python package (pyproject.toml beside this file, scikit-build-core): `pip install tt/cpu_experts` installs
# fnx_cpu/libfnx_cpu.so (fnx_cpu.LIB) built for other PCs: STRATA_PORTABLE (ggml-cpu's AVX2 baseline; the AVX-512
# expert kernels are picked at run time as before) and FNX_ROUTE_MARCH=x86-64-v3 instead of the build host's CPU.
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
add_library(fnx_cpu SHARED fnx_cpu.cpp fnx_route.cpp fnx_qsa.cpp)
# the routing step (fnx_route.cpp) only: AVX2 lanes for its exact f64 dot products; no FMA contraction, so its
# bounds and weights keep the separately rounded products and sums the parity check pins
# FNX_ROUTE_MARCH: the CPU the routing step is compiled for (default: the build host's; x86-64-v3 = AVX2 + FMA for a
# package that runs on other PCs - -ffp-contract=off keeps the rounding the same either way)
set(FNX_ROUTE_MARCH "native" CACHE STRING "-march= of fnx_route.cpp")
set_source_files_properties(fnx_route.cpp PROPERTIES COMPILE_OPTIONS "-march=${FNX_ROUTE_MARCH};-ffp-contract=off")
target_link_libraries(fnx_cpu PRIVATE strata_kernels_cpu)
target_include_directories(fnx_cpu PRIVATE ${STRATA_SRC}/include)
if(SKBUILD)
  # the wheel: the library; the repository's CMakeLists.txt (its `project(strata VERSION ...)` is the engine version
  # fnx.serve_engine reports; fnx.paths.CMAKELISTS reads it through STRATA_CMAKELISTS); data/*.bin (the routing
  # profiles and draft-vocabulary subsets, fnx.paths.STRATA_DATA), which tt-model's code staging does not ship
  # COMPONENT fnx_cpu: the wheel installs only these (pyproject.toml), not ggml's own install rules
  install(TARGETS fnx_cpu LIBRARY DESTINATION fnx_cpu COMPONENT fnx_cpu)
  install(FILES ${STRATA_SRC}/CMakeLists.txt DESTINATION fnx_cpu RENAME strata-CMakeLists.txt COMPONENT fnx_cpu)
  install(DIRECTORY ${STRATA_SRC}/data/ DESTINATION fnx_cpu/data COMPONENT fnx_cpu FILES_MATCHING PATTERN "*.bin")
endif()
