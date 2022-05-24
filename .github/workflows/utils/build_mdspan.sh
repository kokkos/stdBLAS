#!/usr/bin/bash

SRC_PATH=$GITHUB_WORKSPACE/src/mdspan
BUILD_PATH=$GITHUB_WORKSPACE/build/mdspan

# Configure mdspan
cmake \
    -S $SRC_PATH \
    -B $BUILD_PATH \
    -DCMAKE_BUILD_TYPE:STRING=$BUILD_TYPE \
    -DCMAKE_INSTALL_PREFIX:FILEPATH=$INSTALL_PATH \
    -DBUILD_SHARED_LIBS=$SHARED_LIBS

[[ $? -ne 0 ]] && exit 1

# Build mdspan
cmake --build $BUILD_PATH -j $(nproc)

[[ $? -ne 0 ]] && exit 1

# Install mdspan
cmake --install $BUILD_PATH
