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

macro(linalg_add_test name)
  add_executable(${name} ${name}.cpp)
  if(BLAS_FOUND)
    target_link_libraries(${name} linalg GTest::GTest GTest::Main ${BLAS_LIBRARIES})
  else()
    # BLAS_LIBRARIES is literally "FALSE" if the BLAS was not found.
    # Linking against that causes linker errors involving "FALSE.lib".
    # Thus, we exclude BLAS_LIBRARIES completely if the BLAS was not found.
    target_link_libraries(${name} linalg GTest::GTest GTest::Main)
  endif()
  add_test(${name} ${name})
endmacro()

linalg_add_test(abs_sum)
linalg_add_test(add)
linalg_add_test(conjugate_transposed)
linalg_add_test(conjugated)
linalg_add_test(copy)
linalg_add_test(dot)
linalg_add_test(gemm)
linalg_add_test(gemv)
linalg_add_test(givens)
linalg_add_test(idx_abs_max)
# (AMK 6/7/21) Temporarily removing iterator test since it doesn't build
# [ 73%] Building CXX object tests/CMakeFiles/iterator.dir/iterator.cpp.o
# /home/amklinv/stdBLAS/tests/iterator.cpp:12:28: error: 'all' has not been declared in 'std::experimental'
# 12 |   using std::experimental::all;
# linalg_add_test(iterator)
linalg_add_test(matrix_inf_norm)
linalg_add_test(matrix_one_norm)
linalg_add_test(norm2)
linalg_add_test(scale)
linalg_add_test(scaled)
linalg_add_test(swap)
linalg_add_test(symm)
linalg_add_test(transposed)
linalg_add_test(trmm)
