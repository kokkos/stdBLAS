//@HEADER
// ************************************************************************
//
//                        Kokkos v. 4.0
//       Copyright (2022) National Technology & Engineering
//               Solutions of Sandia, LLC (NTESS).
//
// Under the terms of Contract DE-NA0003525 with NTESS,
// the U.S. Government retains certain rights in this software.
//
// Part of Kokkos, under the Apache License v2.0 with LLVM Exceptions.
// See https://kokkos.org/LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// ************************************************************************
//@HEADER

#ifndef LINALG_INCLUDE_EXPERIMENTAL___P1673_BITS_MACROS_HPP_
#define LINALG_INCLUDE_EXPERIMENTAL___P1673_BITS_MACROS_HPP_

#include "__p1673_bits/linalg_config.h"

// Work around a known MSVC issue, that by default
// it always defines __cplusplus as for C++98,
// even if building in a more recent C++ mode.
#ifdef _MSVC_LANG
#define _LINALG_CPLUSPLUS _MSVC_LANG
#else
#define _LINALG_CPLUSPLUS __cplusplus
#endif

#define _LINALG_CXX_STD_14 201402L
#define _LINALG_CXX_STD_17 201703L
#define _LINALG_CXX_STD_20 202002L

#define _LINALG_HAS_CXX_14 (_LINALG_CPLUSPLUS >= _LINALG_CXX_STD_14)
#define _LINALG_HAS_CXX_17 (_LINALG_CPLUSPLUS >= _LINALG_CXX_STD_17)
#define _LINALG_HAS_CXX_20 (_LINALG_CPLUSPLUS >= _LINALG_CXX_STD_20)

static_assert(_LINALG_CPLUSPLUS >= _LINALG_CXX_STD_17, "stdBLAS requires C++17 or later.");

// A sufficiently recent nvc++ comes with <execution>.
// GCC (even 13.1.0) needs TBB, else std::execution::* won't even compile.
// Other compilers like to define __GNUC__ to claim GCC compatibility,
// even if they aren't GCC (and don't have GCC's issue of needing TBB).
#if defined(__NVCOMPILER)
#  define LINALG_HAS_EXECUTION 1
#elif ! defined(__clang__) && ! defined(_MSC_VER) && ! defined(_INTEL_COMPILER) && ! defined(_INTEL_LLMV_COMPILER) && defined(__GNUC__)
#  if defined(LINALG_ENABLE_TBB)
#    define LINALG_HAS_EXECUTION 1
#  endif
#else
#  define LINALG_HAS_EXECUTION 1
#endif

#define P1673_MATRIX_EXTENTS_TEMPLATE_PARAMETERS( MATRIX_NAME ) \
  class SizeType_ ## MATRIX_NAME , \
  ::std::size_t numRows_ ## MATRIX_NAME , \
  ::std::size_t numCols_ ## MATRIX_NAME

#define P1673_MATRIX_TEMPLATE_PARAMETERS( MATRIX_NAME ) \
    class ElementType_ ## MATRIX_NAME , \
    P1673_MATRIX_EXTENTS_TEMPLATE_PARAMETERS( MATRIX_NAME ) , \
    class Layout_ ## MATRIX_NAME , \
    class Accessor_ ## MATRIX_NAME

#define P1673_MATRIX_EXTENTS_PARAMETER( MATRIX_NAME ) \
  extents< \
    SizeType_ ## MATRIX_NAME , \
    numRows_ ## MATRIX_NAME , \
    numCols_ ## MATRIX_NAME \
  >

#define P1673_MATRIX_PARAMETER( MATRIX_NAME ) \
  mdspan< \
    ElementType_ ## MATRIX_NAME , \
    P1673_MATRIX_EXTENTS_PARAMETER( MATRIX_NAME ), \
    Layout_ ## MATRIX_NAME , \
    Accessor_ ## MATRIX_NAME \
  > MATRIX_NAME

#endif //LINALG_INCLUDE_EXPERIMENTAL___P1673_BITS_MACROS_HPP_
