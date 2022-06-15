#!/usr/bin/bash

SRC_PATH=$GITHUB_WORKSPACE/src/kokkos
BUILD_PATH=$GITHUB_WORKSPACE/build/kokkos

# Note: enabling parallel backend, like OpenMP
# gives a chance to detect threading issues, e.g. races
ENABLE_SERIAL=OFF
[ $KOKKOS_BACKEND == 'Serial' ] && ENABLE_SERIAL=ON
ENABLE_OPENMP=OFF
[ $KOKKOS_BACKEND == 'OpenMP' ] && ENABLE_OPENMP=ON
ENABLE_THREADS=OFF
[ $KOKKOS_BACKEND == 'Threads' ] && ENABLE_THREADS=ON

# Configure Kokkos
cmake \
    -S $SRC_PATH \
    -B $BUILD_PATH \
    -DCMAKE_BUILD_TYPE:STRING=$BUILD_TYPE \
    -DCMAKE_INSTALL_PREFIX:FILEPATH=$INSTALL_PATH \
    -DBUILD_SHARED_LIBS=$SHARED_LIBS \
    -DCMAKE_CXX_FLAGS:STRING=-Werror \
    -DKokkos_CXX_STANDARD:STRING=$CXX_STANDARD \
    -DKokkos_ENABLE_COMPLEX_ALIGN:BOOL=OFF \
    -DKokkos_ENABLE_COMPILER_WARNINGS:BOOL=ON \
    -DKokkos_ENABLE_DEPRECATED_CODE_3:BOOL=OFF \
    -DKokkos_ENABLE_TESTS:BOOL=OFF \
    -DKokkos_ENABLE_SERIAL:BOOL=$ENABLE_SERIAL \
    -DKokkos_ENABLE_OPENMP:BOOL=$ENABLE_OPENMP \
    -DKokkos_ENABLE_THREADS:BOOL=$ENABLE_THREADS

[[ $? -ne 0 ]] && exit 2

# Build mdspan
cmake --build $BUILD_PATH -j $(nproc)

[[ $? -ne 0 ]] && exit 1

# Install mdspan
cmake --install $BUILD_PATH
