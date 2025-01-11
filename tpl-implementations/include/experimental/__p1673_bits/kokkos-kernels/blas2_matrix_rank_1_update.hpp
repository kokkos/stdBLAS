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

#ifndef LINALG_TPLIMPLEMENTATIONS_INCLUDE_EXPERIMENTAL___P1673_BITS_KOKKOSKERNELS_BLAS2_MATRIX_RANK_1_UPDATE_HPP_
#define LINALG_TPLIMPLEMENTATIONS_INCLUDE_EXPERIMENTAL___P1673_BITS_KOKKOSKERNELS_BLAS2_MATRIX_RANK_1_UPDATE_HPP_

#include <complex>
#include "signal_kokkos_impl_called.hpp"
#include "static_extent_match.hpp"
#include "triangle.hpp"
#include "parallel_matrix.hpp"

namespace KokkosKernelsSTD {

// Nonsymmetric non-conjugated rank-1 update
// Performs BLAS xGER/xGERU (for real/complex types) A[i,j] += x[i] * y[j]

template<class ExecSpace,
         class ElementType_x,
         std::experimental::extents<>::size_type ext_x,
         class Layout_x,
         class ElementType_y,
         std::experimental::extents<>::size_type ext_y,
         class Layout_y,
         class ElementType_A,
         std::experimental::extents<>::size_type numRows_A,
         std::experimental::extents<>::size_type numCols_A,
         class Layout_A>
void matrix_rank_1_update(kokkos_exec<ExecSpace> &&/* exec */,
  std::experimental::mdspan<ElementType_x, std::experimental::extents<ext_x>, Layout_x,
    std::experimental::default_accessor<ElementType_x>> x,
  std::experimental::mdspan<ElementType_y, std::experimental::extents<ext_y>, Layout_y,
    std::experimental::default_accessor<ElementType_y>> y,
  std::experimental::mdspan<ElementType_A, std::experimental::extents<numRows_A, numCols_A>, Layout_A,
    std::experimental::default_accessor<ElementType_A>> A)
{
  // P1673 constraints (redundant to mdspan extents in the header)
  static_assert(A.rank() == 2);
  static_assert(x.rank() == 1);
  static_assert(y.rank() == 1);

  // P1673 mandates
  static_assert(Impl::static_extent_match(A.static_extent(0), x.static_extent(0)));
  static_assert(Impl::static_extent_match(A.static_extent(1), y.static_extent(0)));

  // P1673 preconditions
  if ( A.extent(0) != x.extent(0) ){
    throw std::runtime_error("KokkosBlas: matrix_rank_1_update: A.extent(0) != x.extent(0)");
  }
  if ( A.extent(1) != y.extent(0) ){
    throw std::runtime_error("KokkosBlas: matrix_rank_1_update: A.extent(1) != y.extent(0)");
  }

  Impl::signal_kokkos_impl_called("matrix_rank1_update");

  // convert mdspans to views and wrap input with original accessors
  const auto x_view = Impl::mdspan_to_view(x);
  const auto y_view = Impl::mdspan_to_view(y);
  auto A_view = Impl::mdspan_to_view(A);
  Impl::ParallelMatrixVisitor v(ExecSpace(), A_view);
  v.for_each_matrix_element(
    KOKKOS_LAMBDA(const auto i, const auto j) {
      A_view(i, j) += x_view(i) * y_view(j);
    });
}

// conjugated(y) specialization dispatched by matrix_rank_1_update_c
template<class ExecSpace,
         class ElementType_x,
         std::experimental::extents<>::size_type ext_x,
         class Layout_x,
         class ElementType_y,
         std::experimental::extents<>::size_type ext_y,
         class Layout_y,
         class ElementType_A,
         std::experimental::extents<>::size_type numRows_A,
         std::experimental::extents<>::size_type numCols_A,
         class Layout_A>
void matrix_rank_1_update(kokkos_exec<ExecSpace> &&/* exec */,
  std::experimental::mdspan<ElementType_x, std::experimental::extents<ext_x>, Layout_x,
    std::experimental::default_accessor<ElementType_x>> x,
  std::experimental::mdspan<ElementType_y, std::experimental::extents<ext_y>, Layout_y,
    std::experimental::linalg::conjugated_accessor<
      std::experimental::default_accessor<ElementType_y>, ElementType_y>> y,
  std::experimental::mdspan<ElementType_A, std::experimental::extents<numRows_A, numCols_A>, Layout_A,
    std::experimental::default_accessor<ElementType_A>> A)
{
  // P1673 constraints (redundant to mdspan extents in the header)
  static_assert(A.rank() == 2);
  static_assert(x.rank() == 1);
  static_assert(y.rank() == 1);

  // P1673 mandates
  static_assert(Impl::static_extent_match(A.static_extent(0), x.static_extent(0)));
  static_assert(Impl::static_extent_match(A.static_extent(1), y.static_extent(0)));

  // P1673 preconditions
  if ( A.extent(0) != x.extent(0) ){
    throw std::runtime_error("KokkosBlas: matrix_rank_1_update: A.extent(0) != x.extent(0)");
  }
  if ( A.extent(1) != y.extent(0) ){
    throw std::runtime_error("KokkosBlas: matrix_rank_1_update: A.extent(1) != y.extent(0)");
  }

  Impl::signal_kokkos_impl_called("matrix_rank1_update");

  auto x_view = Impl::mdspan_to_view(x);
  auto y_view = Impl::mdspan_to_view(y);
  auto A_view = Impl::mdspan_to_view(A);

  using std::experimental::linalg::impl::conj_if_needed;
  Impl::ParallelMatrixVisitor v(ExecSpace(), A_view);
  v.for_each_matrix_element(
    KOKKOS_LAMBDA(const auto i, const auto j) {
      // apply conjugation explicitly (accessor is no longer on the view, see #122)
      A_view(i, j) += x_view(i) * conj_if_needed(y_view(j));
    });
}

// Rank-1 update of a Symmetric matrix
// performs BLAS xSYR/xSPR: A[i,j] += x[i] * x[j]

template<class ExecSpace,
         class ElementType_x,
         std::experimental::extents<>::size_type ext_x,
         class Layout_x,
         class ElementType_A,
         std::experimental::extents<>::size_type numRows_A,
         std::experimental::extents<>::size_type numCols_A,
         class Layout_A,
         class Triangle>
  requires (Impl::is_unique_layout_v<Layout_A, numRows_A, numCols_A>
            or Impl::is_layout_blas_packed_v<Layout_A>)
void symmetric_matrix_rank_1_update(kokkos_exec<ExecSpace> &&exec,
  std::experimental::mdspan<ElementType_x, std::experimental::extents<ext_x>, Layout_x,
    std::experimental::default_accessor<ElementType_x>> x,
  std::experimental::mdspan<ElementType_A, std::experimental::extents<numRows_A, numCols_A>, Layout_A,
    std::experimental::default_accessor<ElementType_A>> A,
  Triangle t)
{
  // P1673 constraints (redundant to mdspan extents in the header)
  static_assert(A.rank() == 2);
  static_assert(x.rank() == 1);
  static_assert(Impl::triangle_layout_match_v<Layout_A, Triangle>);

  // P1673 mandates
  static_assert(Impl::static_extent_match(A.static_extent(0), A.static_extent(1)));
  static_assert(Impl::static_extent_match(A.static_extent(0), x.static_extent(0)));

  // P1673 preconditions
  if ( A.extent(0) != A.extent(1) ){
    throw std::runtime_error("KokkosBlas: symmetric_matrix_rank_1_update: A.extent(0) != A.extent(1)");
  }
  if ( A.extent(0) != x.extent(0) ){
    throw std::runtime_error("KokkosBlas: symmetric_matrix_rank_1_update: A.extent(0) != x.extent(0)");
  }

  Impl::signal_kokkos_impl_called("symmetric_matrix_rank1_update");

  auto x_view = Impl::mdspan_to_view(x);
  auto A_view = Impl::mdspan_to_view(A);
  Impl::ParallelMatrixVisitor v(ExecSpace(), A_view);
  v.for_each_triangle_matrix_element(t,
    KOKKOS_LAMBDA(const auto i, const auto j) {
      A_view(i, j) += x_view(i) * x_view(j);
    });
}


// Rank-1 update of a Hermitian matrix
// performs BLAS xHER/xHPR: A[i,j] += x[i] * conj(x[j])

template<class ExecSpace,
         class ElementType_x,
         std::experimental::extents<>::size_type ext_x,
         class Layout_x,
         class ElementType_A,
         std::experimental::extents<>::size_type numRows_A,
         std::experimental::extents<>::size_type numCols_A,
         class Layout_A,
         class Triangle>
  requires (Impl::is_unique_layout_v<Layout_A, numRows_A, numCols_A>
            or Impl::is_layout_blas_packed_v<Layout_A>)
void hermitian_matrix_rank_1_update(kokkos_exec<ExecSpace> &&exec,
  std::experimental::mdspan<ElementType_x, std::experimental::extents<ext_x>, Layout_x,
    std::experimental::default_accessor<ElementType_x>> x,
  std::experimental::mdspan<ElementType_A, std::experimental::extents<numRows_A, numCols_A>, Layout_A,
    std::experimental::default_accessor<ElementType_A>> A,
  Triangle t)
{
  // P1673 constraints (redundant to mdspan extents in the header)
  static_assert(A.rank() == 2);
  static_assert(x.rank() == 1);
  static_assert(Impl::triangle_layout_match_v<Layout_A, Triangle>);

  // P1673 mandates
  static_assert(Impl::static_extent_match(A.static_extent(0), A.static_extent(1)));
  static_assert(Impl::static_extent_match(A.static_extent(0), x.static_extent(0)));

  // P1673 preconditions
  if ( A.extent(0) != A.extent(1) ){
    throw std::runtime_error("KokkosBlas: hermitian_matrix_rank_1_update: A.extent(0) != A.extent(1)");
  }
  if ( A.extent(0) != x.extent(0) ){
    throw std::runtime_error("KokkosBlas: hermitian_matrix_rank_1_update: A.extent(0) != x.extent(0)");
  }

  Impl::signal_kokkos_impl_called("hermitian_matrix_rank1_update");

  auto x_view = Impl::mdspan_to_view(x);
  auto A_view = Impl::mdspan_to_view(A);

  using std::experimental::linalg::impl::conj_if_needed;
  Impl::ParallelMatrixVisitor v(ExecSpace(), A_view);
  v.for_each_triangle_matrix_element(t,
    KOKKOS_LAMBDA(const auto i, const auto j) {
      A_view(i, j) += x_view(i) * conj_if_needed(x_view(j));
    });
}

} // namespace KokkosKernelsSTD
#endif
