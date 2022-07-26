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

#ifndef LINALG_TPLIMPLEMENTATIONS_INCLUDE_EXPERIMENTAL___P1673_BITS_KOKKOSKERNELS_BLAS2_TRIANGULAR_MATRIX_MATRIX_SOLVE_HPP_
#define LINALG_TPLIMPLEMENTATIONS_INCLUDE_EXPERIMENTAL___P1673_BITS_KOKKOSKERNELS_BLAS2_TRIANGULAR_MATRIX_MATRIX_SOLVE_HPP_

#include "signal_kokkos_impl_called.hpp"
#include "static_extent_match.hpp"
#include "triangle.hpp"

#include <KokkosBlas3_trsm.hpp>

namespace KokkosKernelsSTD {

namespace trimatmatsolve_impl {

template <class Side,
          class Triangle,
          class DiagonalStorage,
          class AViewType,
          class XViewType>
void trsm(Side /*s*/, Triangle /*t*/, DiagonalStorage /*d*/, AViewType A, XViewType X)
{
  const auto side = std::is_same_v<Side,
      std::experimental::linalg::left_side_t> ? "L" : "R";
  const auto uplo = std::is_same_v<Triangle,
      std::experimental::linalg::lower_triangle_t> ? "L" : "U";
  const auto diag = std::is_same_v<DiagonalStorage,
      std::experimental::linalg::explicit_diagonal_t> ? "N" : "U";

  const auto alpha = static_cast<typename XViewType::non_const_value_type>(1.0);
  const auto notranspose = "N";
  KokkosBlas::trsm(side, uplo, notranspose, diag, alpha, A, X);
}

}

// Solve multiple triangular linear systems
// performs BLAS xTRSM

// not-in-place overload
MDSPAN_TEMPLATE_REQUIRES(class ExecSpace,
    class ElementType_A,
    std::experimental::extents<>::size_type numRows_A,
    std::experimental::extents<>::size_type numCols_A,
    class Layout_A,
    class Triangle,
    class DiagonalStorage,
    class ElementType_B,
    std::experimental::extents<>::size_type numRows_B,
    std::experimental::extents<>::size_type numCols_B,
    class Layout_B,
    class ElementType_X,
    std::experimental::extents<>::size_type numRows_X,
    std::experimental::extents<>::size_type numCols_X,
    class Layout_X,
  /* requires */ (Impl::is_unique_layout_v<Layout_X, numRows_X, numCols_X>
        and Impl::is_unique_layout_v<Layout_B, numRows_B, numCols_B>
        and (Impl::is_unique_layout_v<Layout_A, numRows_A, numCols_A>
        or Impl::is_layout_blas_packed_v<Layout_A>)))
void triangular_matrix_matrix_left_solve(kokkos_exec<ExecSpace> &&exec,
  std::experimental::mdspan<ElementType_A, std::experimental::extents<numRows_A, numCols_A>,
    Layout_A, std::experimental::default_accessor<ElementType_A>> A,
  Triangle t,
  DiagonalStorage d,
  std::experimental::mdspan<ElementType_B, std::experimental::extents<numRows_B, numCols_B>,
    Layout_B, std::experimental::default_accessor<ElementType_A>> B,
  std::experimental::mdspan<ElementType_X, std::experimental::extents<numRows_X, numCols_X>,
    Layout_X, std::experimental::default_accessor<ElementType_A>> X)
{
  // P1673 constraints (redundant to mdspan extents in the header)
  static_assert(A.rank() == 2);
  static_assert(X.rank() == 2);
  static_assert(B.rank() == 2);
  static_assert(Impl::triangle_layout_match_v<Layout_A, Triangle>);

  // P1673 mandates
  static_assert(Impl::static_extent_match(A.static_extent(0), A.static_extent(1)));
  static_assert(Impl::static_extent_match(A.static_extent(1), B.static_extent(0)));
  static_assert(Impl::static_extent_match(X.static_extent(0), B.static_extent(0)));
  static_assert(Impl::static_extent_match(X.static_extent(1), B.static_extent(1)));

  // P1673 preconditions
  if ( A.extent(0) != A.extent(1) ){
    throw std::runtime_error("KokkosBlas: triangular_matrix_vector_solve: A.extent(0) != A.extent(1)");
  }
  if ( A.extent(1) != B.extent(0) ){
    throw std::runtime_error("KokkosBlas: triangular_matrix_vector_solve: A.extent(1) != B.extent(0)");
  }
  if ( X.extent(0) != B.extent(0) ){
    throw std::runtime_error("KokkosBlas: triangular_matrix_vector_solve: X.extent(0) != B.extent(0)");
  }
  if ( X.extent(1) != B.extent(1) ){
    throw std::runtime_error("KokkosBlas: triangular_matrix_vector_solve: X.extent(1) != B.extent(1)");
  }

  Impl::signal_kokkos_impl_called("triangular_matrix_matrix_left_solve");

  // convert mdspans to views
  const auto A_view = Impl::mdspan_to_view(A);
  const auto B_view = Impl::mdspan_to_view(B);
  auto X_view = Impl::mdspan_to_view(X);

  Kokkos::deep_copy(X_view, B_view);

  trimatmatsolve_impl::trsm(std::experimental::linalg::left_side, t, d, A_view, X_view);
}

// not-in-place overload
MDSPAN_TEMPLATE_REQUIRES(class ExecSpace,
    class ElementType_A,
    std::experimental::extents<>::size_type numRows_A,
    std::experimental::extents<>::size_type numCols_A,
    class Layout_A,
    class Triangle,
    class DiagonalStorage,
    class ElementType_B,
    std::experimental::extents<>::size_type numRows_B,
    std::experimental::extents<>::size_type numCols_B,
    class Layout_B,
    class ElementType_X,
    std::experimental::extents<>::size_type numRows_X,
    std::experimental::extents<>::size_type numCols_X,
    class Layout_X,
  /* requires */ (Impl::is_unique_layout_v<Layout_X, numRows_X, numCols_X>
        and Impl::is_unique_layout_v<Layout_B, numRows_B, numCols_B>
        and (Impl::is_unique_layout_v<Layout_A, numRows_A, numCols_A>
        or Impl::is_layout_blas_packed_v<Layout_A>)))
void triangular_matrix_matrix_right_solve(kokkos_exec<ExecSpace> &&exec,
  std::experimental::mdspan<ElementType_A, std::experimental::extents<numRows_A, numCols_A>,
    Layout_A, std::experimental::default_accessor<ElementType_A>> A,
  Triangle t,
  DiagonalStorage d,
  std::experimental::mdspan<ElementType_B, std::experimental::extents<numRows_B, numCols_B>,
    Layout_B, std::experimental::default_accessor<ElementType_A>> B,
  std::experimental::mdspan<ElementType_X, std::experimental::extents<numRows_X, numCols_X>,
    Layout_X, std::experimental::default_accessor<ElementType_A>> X)
{
  // P1673 constraints (redundant to mdspan extents in the header)
  static_assert(A.rank() == 2);
  static_assert(X.rank() == 2);
  static_assert(B.rank() == 2);
  static_assert(Impl::triangle_layout_match_v<Layout_A, Triangle>);

  // P1673 mandates
  static_assert(Impl::static_extent_match(A.static_extent(0), A.static_extent(1)));
  static_assert(Impl::static_extent_match(A.static_extent(1), B.static_extent(1)));
  static_assert(Impl::static_extent_match(X.static_extent(0), B.static_extent(0)));
  static_assert(Impl::static_extent_match(X.static_extent(1), B.static_extent(1)));

  // P1673 preconditions
  if ( A.extent(0) != A.extent(1) ){
    throw std::runtime_error("KokkosBlas: triangular_matrix_vector_solve: A.extent(0) != A.extent(1)");
  }
  if ( A.extent(1) != B.extent(1) ){
    throw std::runtime_error("KokkosBlas: triangular_matrix_vector_solve: A.extent(1) != B.extent(1)");
  }
  if ( X.extent(0) != B.extent(0) ){
    throw std::runtime_error("KokkosBlas: triangular_matrix_vector_solve: X.extent(0) != B.extent(0)");
  }
  if ( X.extent(1) != B.extent(1) ){
    throw std::runtime_error("KokkosBlas: triangular_matrix_vector_solve: X.extent(1) != B.extent(1)");
  }

  Impl::signal_kokkos_impl_called("triangular_matrix_matrix_right_solve");

  // convert mdspans to views
  const auto A_view = Impl::mdspan_to_view(A);
  const auto B_view = Impl::mdspan_to_view(B);
  auto X_view = Impl::mdspan_to_view(X);

  Kokkos::deep_copy(X_view, B_view);

  trimatmatsolve_impl::trsm(std::experimental::linalg::right_side, t, d, A_view, X_view);
}

} // namespace KokkosKernelsSTD
#endif
