 /*
//@HEADER
// ************************************************************************
//
//                        Kokkos v. 2.0
//              Copyright (2019) Sandia Corporation
//
// Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
// the U.S. Government retains certain rights in this software.
//
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

#ifndef LINALG_TPLIMPLEMENTATIONS_INCLUDE_EXPERIMENTAL___P1673_BITS_KOKKOSKERNELS_BLAS2_MATRIX_RANK_2_UPDATE_HPP_
#define LINALG_TPLIMPLEMENTATIONS_INCLUDE_EXPERIMENTAL___P1673_BITS_KOKKOSKERNELS_BLAS2_MATRIX_RANK_2_UPDATE_HPP_

#include <complex>
#include "signal_kokkos_impl_called.hpp"
#include "static_extent_match.hpp"
#include "triangle.hpp"
#include "parallel_matrix.hpp"

namespace KokkosKernelsSTD {

// Rank-2 update of a symmetric matrix
// performs BLAS xSYR2/xSPR2: A[i,j] += x[i] * y[j] + y[i] * x[j]

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
         class Layout_A,
         class Triangle>
  requires (Impl::is_unique_layout_v<Layout_A, numRows_A, numCols_A>
            or Impl::is_layout_blas_packed_v<Layout_A>)
void symmetric_matrix_rank_2_update(kokkos_exec<ExecSpace> &&exec,
  std::experimental::mdspan<ElementType_x, std::experimental::extents<ext_x>, Layout_x,
    std::experimental::default_accessor<ElementType_x>> x,
  std::experimental::mdspan<ElementType_y, std::experimental::extents<ext_y>, Layout_y,
    std::experimental::default_accessor<ElementType_y>> y,
  std::experimental::mdspan<ElementType_A, std::experimental::extents<numRows_A, numCols_A>, Layout_A,
    std::experimental::default_accessor<ElementType_A>> A,
  Triangle t)
{
  // P1673 constraints (redundant to mdspan extents in the header)
  static_assert(A.rank() == 2);
  static_assert(x.rank() == 1);
  static_assert(y.rank() == 1);
  static_assert(Impl::triangle_layout_match_v<Layout_A, Triangle>);

  // P1673 mandates
  static_assert(Impl::static_extent_match(A.static_extent(0), A.static_extent(1)));
  static_assert(Impl::static_extent_match(A.static_extent(0), x.static_extent(0)));
  static_assert(Impl::static_extent_match(A.static_extent(0), y.static_extent(0)));

  // P1673 preconditions
  if ( A.extent(0) != A.extent(1) ){
    throw std::runtime_error("KokkosBlas: symmetric_matrix_rank_1_update: A.extent(0) != A.extent(1)");
  }
  if ( A.extent(0) != x.extent(0) ){
    throw std::runtime_error("KokkosBlas: symmetric_matrix_rank_1_update: A.extent(0) != x.extent(0)");
  }
  if ( A.extent(0) != y.extent(0) ){
    throw std::runtime_error("KokkosBlas: symmetric_matrix_rank_1_update: A.extent(0) != y.extent(0)");
  }

  Impl::signal_kokkos_impl_called("symmetric_matrix_rank2_update");

  // convert mdspans to views
  const auto x_view = Impl::mdspan_to_view(x);
  const auto y_view = Impl::mdspan_to_view(y);
  auto A_view = Impl::mdspan_to_view(A);
  Impl::ParallelMatrixVisitor v(ExecSpace(), A_view);
  v.for_each_triangle_matrix_element(t,
    KOKKOS_LAMBDA(const auto i, const auto j) {
      A_view(i, j) += x_view(i) * y_view(j) + y_view(i) * x_view(j);
    });
}

// Rank-2 update of a Hermitian matrix
// performs BLAS xHER2/xHPR2: x[i] * conj(y[j]) + y[i] * conj(x[j])

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
         class Layout_A,
         class Triangle>
  requires (Impl::is_unique_layout_v<Layout_A, numRows_A, numCols_A>
            or Impl::is_layout_blas_packed_v<Layout_A>)
void hermitian_matrix_rank_2_update(kokkos_exec<ExecSpace> &&exec,
  std::experimental::mdspan<ElementType_x, std::experimental::extents<ext_x>, Layout_x,
    std::experimental::default_accessor<ElementType_x>> x,
  std::experimental::mdspan<ElementType_y, std::experimental::extents<ext_y>, Layout_y,
    std::experimental::default_accessor<ElementType_y>> y,
  std::experimental::mdspan<ElementType_A, std::experimental::extents<numRows_A, numCols_A>, Layout_A,
    std::experimental::default_accessor<ElementType_A>> A,
  Triangle t)
{
  // P1673 constraints (redundant to mdspan extents in the header)
  static_assert(A.rank() == 2);
  static_assert(x.rank() == 1);
  static_assert(y.rank() == 1);
  static_assert(Impl::triangle_layout_match_v<Layout_A, Triangle>);

  // P1673 mandates
  static_assert(Impl::static_extent_match(A.static_extent(0), A.static_extent(1)));
  static_assert(Impl::static_extent_match(A.static_extent(0), x.static_extent(0)));
  static_assert(Impl::static_extent_match(A.static_extent(0), y.static_extent(0)));

  // P1673 preconditions
  if ( A.extent(0) != A.extent(1) ){
    throw std::runtime_error("KokkosBlas: hermitian_matrix_rank_1_update: A.extent(0) != A.extent(1)");
  }
  if ( A.extent(0) != x.extent(0) ){
    throw std::runtime_error("KokkosBlas: hermitian_matrix_rank_1_update: A.extent(0) != x.extent(0)");
  }
  if ( A.extent(0) != y.extent(0) ){
    throw std::runtime_error("KokkosBlas: hermitian_matrix_rank_1_update: A.extent(0) != y.extent(0)");
  }

  Impl::signal_kokkos_impl_called("hermitian_matrix_rank2_update");

  // convert mdspans to views
  const auto x_view = Impl::mdspan_to_view(x);
  const auto y_view = Impl::mdspan_to_view(y);
  auto A_view = Impl::mdspan_to_view(A);

  using std::experimental::linalg::impl::conj_if_needed;
  Impl::ParallelMatrixVisitor v(ExecSpace(), A_view);
  v.for_each_triangle_matrix_element(t,
    KOKKOS_LAMBDA(const auto i, const auto j) {
      A_view(i, j) += x_view(i) * conj_if_needed(y_view(j))
                    + y_view(i) * conj_if_needed(x_view(j));
    });
}

} // namespace KokkosKernelsSTD
#endif
