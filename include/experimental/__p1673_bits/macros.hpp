/*
//@HEADER
// ************************************************************************
//
//                        Kokkos v. 2.0
//              Copyright (2019) Sandia Corporation
//
// Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
// the U.S. Government retains certain rights in this software. //
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
//
// 1. Redistributions of source code must retain the above copyright
// notice, this list of conditions and the following disclaimer.
//
// 2. Redistributions in binary form must reproduce the above copyright
// notice, this list of conditions and the following disclaimer in the
// documentation and/or other materials provided with the distribution.
//
// 3. Neither the name of the Corporation nor the names of the
// contributors may be used to endorse or promote products derived from
// this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY SANDIA CORPORATION "AS IS" AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL SANDIA CORPORATION OR THE
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
// LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
// NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
// SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
// Questions? Contact Christian R. Trott (crtrott@sandia.gov)
//
// ************************************************************************
//@HEADER
*/

#ifndef LINALG_INCLUDE_EXPERIMENTAL___P1673_BITS_MACROS_HPP_
#define LINALG_INCLUDE_EXPERIMENTAL___P1673_BITS_MACROS_HPP_

#ifdef _MSVC_LANG
#define _LINALG_CPLUSPLUS _MSVC_LANG
#else
#define _LINALG_CPLUSPLUS __cplusplus
#endif

#define LINALG_CXX_STD_14 201402L
#define LINALG_CXX_STD_17 201703L
#define LINALG_CXX_STD_20 202002L

#define LINALG_HAS_CXX_14 (_LINALG_CPLUSPLUS >= LINALG_CXX_STD_14)
#define LINALG_HAS_CXX_17 (_LINALG_CPLUSPLUS >= LINALG_CXX_STD_17)
#define LINALG_HAS_CXX_20 (_LINALG_CPLUSPLUS >= LINALG_CXX_STD_20)

static_assert(_LINALG_CPLUSPLUS >= LINALG_CXX_STD_17, "stdBLAS requires C++17 or later.");

#if ! defined(__clang__) && ! defined(_MSC_VER) && defined(__GNUC__)
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
  ::std::experimental::extents< \
    SizeType_ ## MATRIX_NAME , \
    numRows_ ## MATRIX_NAME , \
    numCols_ ## MATRIX_NAME \
  >

#define P1673_MATRIX_PARAMETER( MATRIX_NAME ) \
  ::std::experimental::mdspan< \
    ElementType_ ## MATRIX_NAME , \
    P1673_MATRIX_EXTENTS_PARAMETER( MATRIX_NAME ), \
    Layout_ ## MATRIX_NAME , \
    Accessor_ ## MATRIX_NAME \
  > MATRIX_NAME

#endif //LINALG_INCLUDE_EXPERIMENTAL___P1673_BITS_MACROS_HPP_
