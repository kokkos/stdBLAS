find_package(GTest)
if (NOT GTest_FOUND)
  message(STATUS "No installed GTest found, fetching from Github")
  include(FetchContent)
  FetchContent_Declare(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG        release-1.11.0
  )
  # need to set the variables in CACHE due to CMP0077
  set(gtest_disable_pthreads ON CACHE INTERNAL "")
  if(MSVC)
    set(gtest_force_shared_crt ON CACHE INTERNAL "")
  endif()
  FetchContent_GetProperties(googletest)
  if(NOT googletest_POPULATED)
    FetchContent_Populate(googletest)
    add_subdirectory(${googletest_SOURCE_DIR} ${googletest_BINARY_DIR} EXCLUDE_FROM_ALL)
  endif()
  add_library(GTest::Main ALIAS gtest_main)
  add_library(GTest::GTest ALIAS gtest)
endif()

add_subdirectory(native)

if(LINALG_ENABLE_KOKKOS)
  add_subdirectory(kokkos-based)
endif()
