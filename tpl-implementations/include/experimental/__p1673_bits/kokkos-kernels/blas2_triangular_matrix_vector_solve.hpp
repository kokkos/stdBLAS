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

#ifndef LINALG_TPLIMPLEMENTATIONS_INCLUDE_EXPERIMENTAL___P1673_BITS_KOKKOSKERNELS_BLAS2_TRIANGULAR_MATRIX_VECTOR_SOLVE_HPP_
#define LINALG_TPLIMPLEMENTATIONS_INCLUDE_EXPERIMENTAL___P1673_BITS_KOKKOSKERNELS_BLAS2_TRIANGULAR_MATRIX_VECTOR_SOLVE_HPP_

#include "blas3_triangular_matrix_matrix_solve.hpp"

namespace KokkosKernelsSTD {

// Solve a triangular linear system
// performs BLAS xTRSV/xTPSV

// not-in-place overload
MDSPAN_TEMPLATE_REQUIRES(class ExecSpace,
    class ElementType_A,
    std::experimental::extents<>::size_type numRows_A,
    std::experimental::extents<>::size_type numCols_A,
    class Layout_A,
    class Triangle,
    class DiagonalStorage,
    class ElementType_x,
    std::experimental::extents<>::size_type ext_x,
    class Layout_x,
    class ElementType_b,
    std::experimental::extents<>::size_type ext_b,
    class Layout_b,
  /* requires */ (Impl::is_unique_layout_v<Layout_A, numRows_A, numCols_A>
               or Impl::is_layout_blas_packed_v<Layout_A>))
void triangular_matrix_vector_solve(kokkos_exec<ExecSpace> &&exec,
  std::experimental::mdspan<ElementType_A, std::experimental::extents<numRows_A, numCols_A>,
    Layout_A, std::experimental::default_accessor<ElementType_A>> A,
  Triangle t,
  DiagonalStorage d,
  std::experimental::mdspan<ElementType_b, std::experimental::extents<ext_b>,
    Layout_b, std::experimental::default_accessor<ElementType_b>> b,
  std::experimental::mdspan<ElementType_x, std::experimental::extents<ext_x>,
    Layout_x, std::experimental::default_accessor<ElementType_x>> x)
{
  // P1673 constraints (redundant to mdspan extents in the header)
  static_assert(A.rank() == 2);
  static_assert(x.rank() == 1);
  static_assert(b.rank() == 1);
  static_assert(Impl::triangle_layout_match_v<Layout_A, Triangle>);

  // P1673 mandates
  static_assert(Impl::static_extent_match(A.static_extent(0), A.static_extent(1)));
  static_assert(Impl::static_extent_match(A.static_extent(0), x.static_extent(0)));
  static_assert(Impl::static_extent_match(A.static_extent(1), b.static_extent(0)));

  // P1673 preconditions
  if ( A.extent(0) != A.extent(1) ){
    throw std::runtime_error("KokkosBlas: triangular_matrix_vector_solve: A.extent(0) != A.extent(1)");
  }
  if ( A.extent(1) != b.extent(0) ){
    throw std::runtime_error("KokkosBlas: triangular_matrix_vector_solve: A.extent(1) != b.extent(0)");
  }
  if ( A.extent(0) != x.extent(0) ){
    throw std::runtime_error("KokkosBlas: triangular_matrix_vector_solve: A.extent(0) != x.extent(0)");
  }

  Impl::signal_kokkos_impl_called("triangular_matrix_vector_solve");

  // convert mdspans to views
  const auto A_view = Impl::mdspan_to_view(A);
  const auto b_view = Impl::mdspan_to_view(b);
  auto x_view = Impl::mdspan_to_view(x);

  // using in-place routine on x=b
  Kokkos::deep_copy(x_view, b_view);

  // promote x to Nx1 multivector to use KokkosBlas::trsm()
  // (trsv is not implemented in KokkosBlas)
  using x_view_type = decltype(x_view);
  using x_scalar_type = typename x_view_type::non_const_value_type;
  Kokkos::View<std::add_pointer_t<std::add_pointer_t<x_scalar_type>>,
    typename x_view_type::array_layout,
    typename x_view_type::device_type,
    typename x_view_type::memory_traits> X_view(x_view.data(), x_view.extent(0), 1);

  trimatmatsolve_impl::trsm(std::experimental::linalg::left_side, t, d, A_view, X_view);
}

} // namespace KokkosKernelsSTD
#endif
