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

#ifndef LINALG_TPLIMPLEMENTATIONS_INCLUDE_EXPERIMENTAL___P1673_BITS_KOKKOSKERNELS_BLAS2_MATRIX_RANK_1_UPDATE_HPP_
#define LINALG_TPLIMPLEMENTATIONS_INCLUDE_EXPERIMENTAL___P1673_BITS_KOKKOSKERNELS_BLAS2_MATRIX_RANK_1_UPDATE_HPP_

#include <complex>
#include "signal_kokkos_impl_called.hpp"

namespace KokkosKernelsSTD {

namespace Impl {

// manages parallel execution of independent action
// called like action(i, j) for each matrix element A(i, j)
template <typename ExecSpace, typename MatrixType>
class ParallelMatrixVisitor {
public:
  KOKKOS_INLINE_FUNCTION ParallelMatrixVisitor(ExecSpace &&exec_in, MatrixType &A_in):
    exec(exec_in), A(A_in), ext0(A.extent(0)), ext1(A.extent(1))
  {}

  template <typename ActionType>
  KOKKOS_INLINE_FUNCTION
  void for_each_matrix_element(ActionType action) {
    if (ext0 > ext1) { // parallel rows
      Kokkos::parallel_for(Kokkos::RangePolicy(exec, 0, ext0),
        KOKKOS_LAMBDA(const auto i) {
          using idx_type = std::remove_const_t<decltype(i)>;
          for (idx_type j = 0; j < ext1; ++j) {
            action(i, j);
          }
        });
    } else { // parallel columns
      Kokkos::parallel_for(Kokkos::RangePolicy(exec, 0, ext1),
        KOKKOS_LAMBDA(const auto j) {
          using idx_type = std::remove_const_t<decltype(j)>;
          for (idx_type i = 0; i < ext0; ++i) {
            action(i, j);
          }
        });
    }
    exec.fence();
  }

private:
  const ExecSpace &exec;
  MatrixType &A;
  const size_t ext0;
  const size_t ext1;
};

// This version of conj_if_needed() also handles Kokkos::complex<T>
template <class T>
KOKKOS_INLINE_FUNCTION
T conj_if_needed(const T &value)
{
  return value;
};

template <class T>
KOKKOS_INLINE_FUNCTION
auto conj_if_needed(const Kokkos::complex<T> &value)
{
  return Kokkos::conj(value);
};

template <class T>
KOKKOS_INLINE_FUNCTION
auto conj_if_needed(const std::complex<T> &value)
{
  return std::conj(value);
};

} // namespace Impl

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
  // constraints
  static_assert(A.rank() == 2);
  static_assert(x.rank() == 1);
  static_assert(y.rank() == 1);

  // preconditions
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
    std::experimental::linalg::accessor_conjugate<
      std::experimental::default_accessor<ElementType_y>, ElementType_y>> y,
  std::experimental::mdspan<ElementType_A, std::experimental::extents<numRows_A, numCols_A>, Layout_A,
    std::experimental::default_accessor<ElementType_A>> A)
{
  // constraints
  static_assert(A.rank() == 2);
  static_assert(x.rank() == 1);
  static_assert(y.rank() == 1);

  // preconditions
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

  Impl::ParallelMatrixVisitor v(ExecSpace(), A_view);
  v.for_each_matrix_element(
    KOKKOS_LAMBDA(const auto i, const auto j) {
      // apply conjugation explicitly (accessor is no longer on the view, see #122)
      A_view(i, j) += x_view(i) * Impl::conj_if_needed(y_view(j));
    });
}

} // namespace KokkosKernelsSTD
#endif
