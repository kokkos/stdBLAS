# Reference Implementation for BLAS in the C++ standard

## Requirements

  - CMake >= 3.12 (requirement of mdarray)
  - C++14 or greater compiler (requirement of mdspan)

## Build instructions

1. Download and install googletest
   - https://github.com/google/googletest
   - (master appears to work)
2. Download and install mdspan:
   - git@github.com:kokkos/mdspan.git
3. Download and install mdarray:
   - git@github.com:dhollman/mdarray.git
   - Set `CMAKE_PREFIX_PATH` to where you installed mdspan.
