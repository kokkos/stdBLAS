#!/usr/bin/bash

SRC_PATH=$GITHUB_WORKSPACE/src/kokkos-kernels
BUILD_PATH=$GITHUB_WORKSPACE/build/kokkos-kernels

ENABLE_KK_ETI=OFF

# Configure Kokkos-Kernels
cmake \
    -S $SRC_PATH \
    -B $BUILD_PATH \
    -DCMAKE_BUILD_TYPE:STRING=$BUILD_TYPE \
    -DCMAKE_INSTALL_PREFIX:FILEPATH=$INSTALL_PATH \
    -DBUILD_SHARED_LIBS=$SHARED_LIBS \
    -DKokkosKernels_ENABLE_TESTS:BOOL=OFF \
    -DKokkosKernels_ENABLE_DOCS:BOOL=OFF \
    -DKokkosKernels_ENABLE_TESTS_AND_PERFSUITE:BOOL=OFF \
    -DKokkosKernels_ENABLE_EXPERIMENTAL:BOOL=ON \
    -DKokkosKernels_INST_DOUBLE:BOOL=$ENABLE_KK_ETI \
    -DKokkosKernels_INST_FLOAT:BOOL=OFF \
    -DKokkosKernels_INST_COMPLEX_FLOAT:BOOL=OFF \
    -DKokkosKernels_INST_COMPLEX_DOUBLE:BOOL=OFF \
    -DKokkosKernels_INST_ORDINAL_INT:BOOL=$ENABLE_KK_ETI \
    -DKokkosKernels_INST_ORDINAL_INT64_T:BOOL=OFF \
    -DKokkosKernels_INST_OFFSET_INT:BOOL=$ENABLE_KK_ETI \
    -DKokkosKernels_INST_OFFSET_SIZE_T:BOOL=OFF \
    -DKokkosKernels_INST_LAYOUTLEFT:BOOL=$ENABLE_KK_ETI \
    -DKokkosKernels_INST_LAYOUTRIGHT:BOOL=OFF
[[ $? -ne 0 ]] && exit 1


# Build Kokkos-Kernels
cmake --build $GITHUB_WORKSPACE/build/kokkos-kernels -j $(nproc)

[[ $? -ne 0 ]] && exit 1

# Install Kokkos-Kernels
cmake --install $GITHUB_WORKSPACE/build/kokkos-kernels
