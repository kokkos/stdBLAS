
cmake_minimum_required(VERSION 3.12)
project(LinAlg
  VERSION 0.0.1
  LANGUAGES CXX
)

################################################################################

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

################################################################################

option(LINALG_ENABLE_TESTS "Enable tests." Off)
option(LINALG_ENABLE_EXAMPLES "Build examples." Off)
#option(LINALG_ENABLE_BENCHMARKS "Enable benchmarks." Off)
#option(LINALG_ENABLE_COMP_BENCH "Enable compilation benchmarks." Off)

# Option to override which C++ standard to use
set(LINALG_CXX_STANDARD DETECT CACHE STRING "Override the default CXX_STANDARD to compile with.")
set_property(CACHE LINALG_CXX_STANDARD PROPERTY STRINGS DETECT 20)

option(LINALG_ENABLE_CONCEPTS "Try to enable concepts support by giving extra flags." On)

################################################################################

# Make sure that the compiler supports C++20
if(LINALG_CXX_STANDARD STREQUAL "20")
  if("cxx_std_20" IN_LIST CMAKE_CXX_COMPILE_FEATURES)
    message(STATUS "Using C++20 standard")
    set(CMAKE_CXX_STANDARD 20)
  else()
    message(FATAL_ERROR "Requested LINALG_CXX_STANDARD \"20\" not supported by provided C++ compiler")
  endif()
else()
  if("cxx_std_20" IN_LIST CMAKE_CXX_COMPILE_FEATURES)
    set(CMAKE_CXX_STANDARD 20)
    message(STATUS "Detected support for C++20 standard")
  else()
    message(FATAL_ERROR "Cannot detect CXX_STANDARD of C++20")
  endif()
endif()

################################################################################

if(LINALG_ENABLE_CONCEPTS)
  if(CMAKE_CXX_STANDARD STREQUAL "20")
    include(CheckCXXCompilerFlag)
    CHECK_CXX_COMPILER_FLAG("-fconcepts" COMPILER_SUPPORTS_FCONCEPTS)
    if(COMPILER_SUPPORTS_FCONCEPTS)
      message(STATUS "-- Using \"-fconcepts\" to enable concepts support")
      set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fconcepts")
    else()
      CHECK_CXX_COMPILER_FLAG("-fconcepts-ts" COMPILER_SUPPORTS_FCONCEPTS_TS)
      if(COMPILER_SUPPORTS_FCONCEPTS)
        message(STATUS "-- Using \"-fconcepts-ts\" to enable concepts support")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fconcepts-ts")
      endif()
    endif()
    # Otherwise, it's possible that the compiler supports concepts without flags,
    # but if it doesn't, they just won't be used, which is fine
  endif()
endif()

################################################################################

find_package(mdspan QUIET)
if (NOT mdspan_FOUND)
  message(STATUS "No installed mdspan found, fetching from Github")
  include(FetchContent)
  FetchContent_Declare(
    mdspan
    GIT_REPOSITORY https://github.com/kokkos/mdspan.git
    GIT_TAG        stable
  )
  FetchContent_GetProperties(mdspan)
  if(NOT mdspan_POPULATED)
    FetchContent_Populate(mdspan)
    add_subdirectory(${mdspan_SOURCE_DIR} ${mdspan_BINARY_DIR} EXCLUDE_FROM_ALL)
  endif()
endif()

find_package(BLAS)
option(LINALG_ENABLE_BLAS
  "Assume that we are linking with a BLAS library."
  ${BLAS_FOUND})

################################################################################

CONFIGURE_FILE(include/experimental/__p1673_bits/linalg_config.h.in
  include/experimental/__p1673_bits/linalg_config.h)
include_directories(${CMAKE_CURRENT_BINARY_DIR}/include/experimental)
message(STATUS "Build include directory: ${CMAKE_CURRENT_BINARY_DIR}/include/experimental")

################################################################################

add_library(linalg INTERFACE)
add_library(std::linalg ALIAS linalg)

target_link_libraries(linalg INTERFACE std::mdspan)
if(LINALG_ENABLE_KOKKOS)
  find_package(Kokkos)
  find_package(KokkosKernels)
  target_link_libraries(linalg INTERFACE Kokkos::kokkos)
  target_link_libraries(linalg INTERFACE Kokkos::kokkoskernels)
  target_include_directories(linalg INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/tpl-implementations/include>
  )
endif()

target_include_directories(linalg INTERFACE
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)

################################################################################

install(TARGETS linalg EXPORT linalgTargets
  INCLUDES DESTINATION include
)

install(EXPORT linalgTargets
    FILE linalgTargets.cmake
    NAMESPACE std::
    DESTINATION lib/cmake/stdBLAS
)

export(TARGETS linalg
    NAMESPACE std::
    FILE linalgTargets.cmake
)

install(DIRECTORY include/experimental DESTINATION include)
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/include/experimental/__p1673_bits/linalg_config.h
    DESTINATION include/experimental/__p1673_bits
)

include(CMakePackageConfigHelpers)
configure_package_config_file(cmake/LinAlgConfig.cmake.in
  ${CMAKE_CURRENT_BINARY_DIR}/LinAlgConfig.cmake
  INSTALL_DESTINATION cmake
)
write_basic_package_version_file(${CMAKE_CURRENT_BINARY_DIR}/LinAlgConfigVersion.cmake
  COMPATIBILITY SameMajorVersion
)

install(FILES ${CMAKE_CURRENT_BINARY_DIR}/LinAlgConfig.cmake ${CMAKE_CURRENT_BINARY_DIR}/LinAlgConfigVersion.cmake
    DESTINATION lib/cmake/stdBLAS
)

################################################################################

if(LINALG_ENABLE_TESTS)
  enable_testing()
  add_subdirectory(tests)
  add_subdirectory(compilation_tests)
endif()

if(LINALG_ENABLE_EXAMPLES)
 add_subdirectory(examples)
endif()

#if(LINALG_ENABLE_BENCHMARKS)
# add_subdirectory(benchmarks)
#endif()
#
#if(LINALG_ENABLE_COMP_BENCH)
#  add_subdirectory(comp_bench)
#endif()
