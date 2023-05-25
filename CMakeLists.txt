
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
set_property(CACHE LINALG_CXX_STANDARD PROPERTY STRINGS DETECT 14 17 20 23)

option(LINALG_ENABLE_CONCEPTS "Try to enable concepts support by giving extra flags." On)
option(LINALG_ENABLE_ATOMIC_REF "Try to enable atomic_ref support" OFF)

################################################################################

# Decide on the standard to use
if(LINALG_CXX_STANDARD STREQUAL "17")
  if("cxx_std_17" IN_LIST CMAKE_CXX_COMPILE_FEATURES)
    message(STATUS "Using C++17 standard")
    set(CMAKE_CXX_STANDARD 17)
  else()
    message(FATAL_ERROR "Requested LINALG_CXX_STANDARD \"17\" not supported by provided C++ compiler")
  endif()
elseif(LINALG_CXX_STANDARD STREQUAL "14")
  if("cxx_std_14" IN_LIST CMAKE_CXX_COMPILE_FEATURES)
    message(STATUS "Using C++14 standard")
    set(CMAKE_CXX_STANDARD 14)
  else()
    message(FATAL_ERROR "Requested LINALG_CXX_STANDARD \"14\" not supported by provided C++ compiler")
  endif()
elseif(LINALG_CXX_STANDARD STREQUAL "20")
  if("cxx_std_20" IN_LIST CMAKE_CXX_COMPILE_FEATURES)
    message(STATUS "Using C++20 standard")
    set(CMAKE_CXX_STANDARD 20)
  else()
    message(FATAL_ERROR "Requested LINALG_CXX_STANDARD \"20\" not supported by provided C++ compiler")
  endif()
elseif(LINALG_CXX_STANDARD STREQUAL "23")
  if("cxx_std_23" IN_LIST CMAKE_CXX_COMPILE_FEATURES)
    message(STATUS "Using C++23 standard")
    set(CMAKE_CXX_STANDARD 23)
  else()
    message(FATAL_ERROR "Requested LINALG_CXX_STANDARD \"23\" not supported by provided C++ compiler")
  endif()
else()
  if("cxx_std_23" IN_LIST CMAKE_CXX_COMPILE_FEATURES)
    set(CMAKE_CXX_STANDARD 23)
    message(STATUS "Detected support for C++23 standard")
  elseif("cxx_std_20" IN_LIST CMAKE_CXX_COMPILE_FEATURES)
    set(CMAKE_CXX_STANDARD 20)
    message(STATUS "Detected support for C++20 standard")
  elseif("cxx_std_17" IN_LIST CMAKE_CXX_COMPILE_FEATURES)
    set(CMAKE_CXX_STANDARD 17)
    message(STATUS "Detected support for C++17 standard")
  elseif("cxx_std_14" IN_LIST CMAKE_CXX_COMPILE_FEATURES)
    set(CMAKE_CXX_STANDARD 14)
    message(STATUS "Detected support for C++14 standard")
  else()
    message(FATAL_ERROR "Cannot detect CXX_STANDARD of C++14 or newer.")
  endif()
endif()

################################################################################

if(LINALG_ENABLE_CONCEPTS)
  if(CMAKE_CXX_STANDARD GREATER_EQUAL 20)
    include(CheckCXXCompilerFlag)
    check_cxx_compiler_flag("-fconcepts" COMPILER_SUPPORTS_FCONCEPTS)
    if(COMPILER_SUPPORTS_FCONCEPTS)
      message(STATUS "Using \"-fconcepts\" to enable concepts support")
      set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fconcepts")
    else()
      check_cxx_compiler_flag("-fconcepts-ts" COMPILER_SUPPORTS_FCONCEPTS_TS)
      if(COMPILER_SUPPORTS_FCONCEPTS)
        message(STATUS "Using \"-fconcepts-ts\" to enable concepts support")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fconcepts-ts")
      endif()
    endif()
    # Otherwise, it's possible that the compiler supports concepts without flags,
    # but if it doesn't, they just won't be used, which is fine
  endif()
endif()

if(LINALG_ENABLE_ATOMIC_REF)
  # check_linker_flag requires CMake 3.18.
  set(LINK_OPTIONS "${LINK_OPTIONS} -latomic")
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

find_package(KokkosKernels)
option(LINALG_ENABLE_KOKKOS
  "Enable Kokkos-based implementation. Default: autodetect Kokkos installation."
  ${KokkosKernels_FOUND})
if(LINALG_ENABLE_KOKKOS)
  find_package(Kokkos REQUIRED)
  find_package(KokkosKernels REQUIRED)
endif()

find_package(TBB)
option(LINALG_ENABLE_TBB
  "Enable Threaded Building Blocks for tests. Default: autodetect TBB installation."
  ${TBB_FOUND}
)
if(LINALG_ENABLE_TBB)
  find_package(TBB REQUIRED)
endif()

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
  target_link_libraries(linalg INTERFACE Kokkos::kokkos)
  target_link_libraries(linalg INTERFACE Kokkos::kokkoskernels)
  target_include_directories(linalg INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/tpl-implementations/include>
  )
endif()

if(LINALG_ENABLE_TBB)
  target_link_libraries(linalg INTERFACE TBB::tbb)
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
    DESTINATION cmake
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
    DESTINATION cmake
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
