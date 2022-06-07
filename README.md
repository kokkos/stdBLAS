# P1673 reference implementation

This is a reference implementation of P1673,
"A free function linear algebra interface based on the BLAS."
You can find the latest submitted revision of P1673
[at this URL](https://wg21.link/p1673).

## Requirements

  - CMake >= 3.17 (earlier versions may work, but are not tested)
  - C++ build environment that supports C++17 or greater

## Tested compilers

We run github's automated tests on every pull request.
Automated tests use "ubuntu-latest",
which presumably defaults to a fairly new GCC.
Other compilers, including MSVC 2019, have been tested in the past.

## Brief build instructions

1. Download and install googletest (GTest)
   - https://github.com/google/googletest
2. Download and install mdspan:
   - git@github.com:kokkos/mdspan.git
3. Run CMake, pointing it to your googletest and mdspan install locations
   - If you want to build tests, set LINALG_ENABLE_TESTS=ON
   - If you want to build examples, set LINALG_ENABLE_EXAMPLES=ON
   - If you have a BLAS installation, set LINALG_ENABLE_BLAS=ON.
     BLAS support is currently experimental.
4. Build and install as usual
5. If you enabled tests, use "ctest" to run them

## More detailed MSVC build instructions

Be sure to build mdspan and googletest in the Release configuration before installing.

The following CMake options are known to work:

- mdspan_DIR=${MDSPAN_INSTALL_DIR}\lib\cmake\mdspan
  (where MDSPAN_INSTALL_DIR is the path to your mdspan installation)
- GTEST_INCLUDE_DIR=${GTEST_INSTALL_DIR}\include
  (where GTEST_INSTALL_DIR is the path to your googletest installation)
- GTEST_LIBRARY=${GTEST_INSTALL_DIR}\lib\gtest.lib
- GTEST_MAIN_LIBRARY=${GTEST_INSTALL_DIR}\lib\gtest_main.lib

When building tests, for all CMAKE_CXX_FLAGS_* options,
you might need to change "/MD" to "/MT", depending on how googletest was built.
