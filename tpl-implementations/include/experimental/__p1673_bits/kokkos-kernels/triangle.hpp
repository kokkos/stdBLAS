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

#ifndef LINALG_TPLIMPLEMENTATIONS_INCLUDE_EXPERIMENTAL___P1673_BITS_KOKKOSKERNELS_TRIANGLE_UTILS_HPP_
#define LINALG_TPLIMPLEMENTATIONS_INCLUDE_EXPERIMENTAL___P1673_BITS_KOKKOSKERNELS_TRIANGLE_UTILS_HPP_

namespace KokkosKernelsSTD {
namespace Impl {

// Note: phrase it simply and the same as in specification ("has unique layout")
template <typename Layout,
          std::experimental::extents<>::size_type numRows,
          std::experimental::extents<>::size_type numCols>
constexpr bool is_unique_layout_v = Layout::template mapping<
    std::experimental::extents<numRows, numCols> >::is_always_unique();

template <typename Layout>
struct is_layout_blas_packed: public std::false_type {};

template <typename Triangle, typename StorageOrder>
struct is_layout_blas_packed<
  std::experimental::linalg::layout_blas_packed<Triangle, StorageOrder>>:
    public std::true_type {};

template <typename Layout>
constexpr bool is_layout_blas_packed_v = is_layout_blas_packed<Layout>::value;

// Note: will only signal failure for layout_blas_packed with different triangle
template <typename Layout, typename Triangle>
struct triangle_layout_match: public std::true_type {};

template <typename StorageOrder, typename Triangle1, typename Triangle2>
struct triangle_layout_match<
  std::experimental::linalg::layout_blas_packed<Triangle1, StorageOrder>,
  Triangle2>
{
  static constexpr bool value = std::is_same_v<Triangle1, Triangle2>;
};

template <typename Layout, typename Triangle>
constexpr bool triangle_layout_match_v = triangle_layout_match<Layout, Triangle>::value;

} // namespace Impl
} // namespace KokkosKernelsSTD
#endif
