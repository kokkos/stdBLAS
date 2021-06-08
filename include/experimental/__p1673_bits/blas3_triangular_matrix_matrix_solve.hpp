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

#ifndef LINALG_INCLUDE_EXPERIMENTAL___P1673_BITS_BLAS3_TRIANGULAR_MATRIX_MATRIX_SOLVE_HPP_
#define LINALG_INCLUDE_EXPERIMENTAL___P1673_BITS_BLAS3_TRIANGULAR_MATRIX_MATRIX_SOLVE_HPP_

namespace std {
namespace experimental {
inline namespace __p1673_version_0 {
namespace linalg {

namespace {

template<class ElementType_A,
         extents<>::size_type numRows_A,
         extents<>::size_type numCols_A,
         class Layout_A,
         class Accessor_A,
         class DiagonalStorage,
         class ElementType_B,
         extents<>::size_type numRows_B,
         extents<>::size_type numCols_B,
         class Layout_B,
         class Accessor_B,
         class ElementType_X,
         extents<>::size_type numRows_X,
         extents<>::size_type numCols_X,
         class Layout_X,
         class Accessor_X>
void trsm_upper_triangular_left_side(
  std::experimental::basic_mdspan<ElementType_A, std::experimental::extents<numRows_A, numCols_A>, Layout_A, Accessor_A> A,
  DiagonalStorage d,
  std::experimental::basic_mdspan<ElementType_B, std::experimental::extents<numRows_B, numCols_B>, Layout_B, Accessor_B> B,
  std::experimental::basic_mdspan<ElementType_X, std::experimental::extents<numRows_X, numCols_X>, Layout_X, Accessor_X> X)
{
  constexpr bool explicit_diagonal =
    std::is_same_v<DiagonalStorage, explicit_diagonal_t>;
  using size_type = typename extents<>::size_type;

  const size_type A_num_rows = A.extent(0);
  const size_type B_num_cols = B.extent(1);

  for (size_type k = 0; k < B_num_cols; ++k) {
    // One advantage of using signed index types is that you can write
    // descending loops with zero-based indices.
    for (size_type i = A_num_rows - 1; i >= 0; --i) {
      // TODO this would be a great opportunity for an implementer to
      // add value, by accumulating in extended precision (or at least
      // in a type with the max precision of X and B).
      using sum_type = decltype (B(i,k) - A(0,0) * X(0,0));
      //using sum_type = typename out_object_t::element_type;
      const sum_type t (B(i,k));
      for (size_type j = i + 1; j < A_num_rows; ++j) {
        t = t - A(i,j) * X(j,k);
      }
      if constexpr (explicit_diagonal) {
        X(i,k) = t / A(i,i);
      }
      else {
        X(i,k) = t;
      }
    }
  }
}

template<class ElementType_A,
         extents<>::size_type numRows_A,
         extents<>::size_type numCols_A,
         class Layout_A,
         class Accessor_A,
         class DiagonalStorage,
         class ElementType_B,
         extents<>::size_type numRows_B,
         extents<>::size_type numCols_B,
         class Layout_B,
         class Accessor_B,
         class ElementType_X,
         extents<>::size_type numRows_X,
         extents<>::size_type numCols_X,
         class Layout_X,
         class Accessor_X>
void trsm_lower_triangular_left_side(
  std::experimental::basic_mdspan<ElementType_A, std::experimental::extents<numRows_A, numCols_A>, Layout_A, Accessor_A> A,
  DiagonalStorage d,
  std::experimental::basic_mdspan<ElementType_B, std::experimental::extents<numRows_B, numCols_B>, Layout_B, Accessor_B> B,
  std::experimental::basic_mdspan<ElementType_X, std::experimental::extents<numRows_X, numCols_X>, Layout_X, Accessor_X> X)
{
  constexpr bool explicit_diagonal =
    std::is_same_v<DiagonalStorage, explicit_diagonal_t>;
  using size_type = typename extents<>::size_type;

  const size_type A_num_rows = A.extent(0);
  const size_type B_num_cols = B.extent(1);

  for (size_type k = 0; k < B_num_cols; ++k) {
    for (size_type i = 0; i < A_num_rows; ++i) {
      // TODO this would be a great opportunity for an implementer to
      // add value, by accumulating in extended precision (or at least
      // in a type with the max precision of X and B).
      const ElementType_X t (B(i,k));
      for (size_type j = 0; j < i; ++j) {
        t = t - A(i,j) * X(j,k);
      }
      if constexpr (explicit_diagonal) {
        X(i,k) = t / A(i,i);
      }
      else {
        X(i,k) = t;
      }
    }
  }
}

template<class ElementType_A,
         extents<>::size_type numRows_A,
         extents<>::size_type numCols_A,
         class Layout_A,
         class Accessor_A,
         class DiagonalStorage,
         class ElementType_B,
         extents<>::size_type numRows_B,
         extents<>::size_type numCols_B,
         class Layout_B,
         class Accessor_B,
         class ElementType_X,
         extents<>::size_type numRows_X,
         extents<>::size_type numCols_X,
         class Layout_X,
         class Accessor_X>
void trsm_upper_triangular_right_side(
  std::experimental::basic_mdspan<ElementType_A, std::experimental::extents<numRows_A, numCols_A>, Layout_A, Accessor_A> A,
  DiagonalStorage d,
  std::experimental::basic_mdspan<ElementType_B, std::experimental::extents<numRows_B, numCols_B>, Layout_B, Accessor_B> B,
  std::experimental::basic_mdspan<ElementType_X, std::experimental::extents<numRows_X, numCols_X>, Layout_X, Accessor_X> X)
{
  constexpr bool explicit_diagonal =
    std::is_same_v<DiagonalStorage, explicit_diagonal_t>;
  using size_type = typename extents<>::size_type;

  const size_type B_num_rows = B.extent(0);
  const size_type A_num_cols = A.extent(1);

  for (size_type i = 0; i < B_num_rows; ++i) {
    for (size_type j = 0; j < A_num_cols; ++j) {
      using sum_type = decltype (B(i,j) - A(0,0) * X(0,0));
      const sum_type t (B(i,j));
      for (size_type k = 0; k < j; ++j) {
        t = t - X(i,k) * A(k,j);
      }
      if constexpr (explicit_diagonal) {
        X(i,j) = t / A(j,j);
      }
      else {
        X(i,j) = t;
      }
    }
  }
}

template<class ElementType_A,
         extents<>::size_type numRows_A,
         extents<>::size_type numCols_A,
         class Layout_A,
         class Accessor_A,
         class DiagonalStorage,
         class ElementType_B,
         extents<>::size_type numRows_B,
         extents<>::size_type numCols_B,
         class Layout_B,
         class Accessor_B,
         class ElementType_X,
         extents<>::size_type numRows_X,
         extents<>::size_type numCols_X,
         class Layout_X,
         class Accessor_X>
void trsm_lower_triangular_right_side(
  std::experimental::basic_mdspan<ElementType_A, std::experimental::extents<numRows_A, numCols_A>, Layout_A, Accessor_A> A,
  DiagonalStorage d,
  std::experimental::basic_mdspan<ElementType_B, std::experimental::extents<numRows_B, numCols_B>, Layout_B, Accessor_B> B,
  std::experimental::basic_mdspan<ElementType_X, std::experimental::extents<numRows_X, numCols_X>, Layout_X, Accessor_X> X)
{
  constexpr bool explicit_diagonal =
    std::is_same_v<DiagonalStorage, explicit_diagonal_t>;
  using size_type = typename extents<>::size_type;

  const size_type B_num_rows = B.extent(0);
  const size_type A_num_rows = A.extent(0);
  const size_type A_num_cols = A.extent(1);

  for (size_type i = 0; i < B_num_rows; ++i) {
    for (size_type j = 0; j < A_num_cols; ++j) {
      using sum_type = decltype (B(i,j) - A(0,0) * X(0,0));
      const sum_type t (B(i,j));
      for (size_type k = j + 1; k < A_num_rows; ++j) {
        t = t - X(i,k) * A(k,j);
      }
      if constexpr (explicit_diagonal) {
        X(i,j) = t / A(j,j);
      }
      else {
        X(i,j) = t;
      }
    }
  }
}

} // end anonymous namespace

template<class ElementType_A,
         extents<>::size_type numRows_A,
         extents<>::size_type numCols_A,
         class Layout_A,
         class Accessor_A,
         class Triangle,
         class DiagonalStorage,
         class Side,
         class ElementType_B,
         extents<>::size_type numRows_B,
         extents<>::size_type numCols_B,
         class Layout_B,
         class Accessor_B,
         class ElementType_X,
         extents<>::size_type numRows_X,
         extents<>::size_type numCols_X,
         class Layout_X,
         class Accessor_X>
void triangular_matrix_matrix_solve(
  std::experimental::basic_mdspan<ElementType_A, std::experimental::extents<numRows_A, numCols_A>, Layout_A, Accessor_A> A,
  Triangle t,
  DiagonalStorage d,
  Side s,
  std::experimental::basic_mdspan<ElementType_B, std::experimental::extents<numRows_B, numCols_B>, Layout_B, Accessor_B> B,
  std::experimental::basic_mdspan<ElementType_X, std::experimental::extents<numRows_X, numCols_X>, Layout_X, Accessor_X> X)
{
  if (std::is_same_v<Side, left_side_t>) {
    if (std::is_same_v<Triangle, lower_triangle_t>) {
      trsm_lower_triangular_left_side (A, d, B, X);
    }
    else {
      trsm_upper_triangular_left_side (A, d, B, X);
    }
  }
  else {
    if (std::is_same_v<Triangle, lower_triangle_t>) {
      trsm_lower_triangular_right_side (A, d, B, X);
    }
    else {
      trsm_upper_triangular_right_side (A, d, B, X);
    }
  }
}

template<class ExecutionPolicy,
         class ElementType_A,
         extents<>::size_type numRows_A,
         extents<>::size_type numCols_A,
         class Layout_A,
         class Accessor_A,
         class Triangle,
         class DiagonalStorage,
         class Side,
         class ElementType_B,
         extents<>::size_type numRows_B,
         extents<>::size_type numCols_B,
         class Layout_B,
         class Accessor_B,
         class ElementType_X,
         extents<>::size_type numRows_X,
         extents<>::size_type numCols_X,
         class Layout_X,
         class Accessor_X>
void triangular_matrix_matrix_solve(
  ExecutionPolicy&& /* exec */,
  std::experimental::basic_mdspan<ElementType_A, std::experimental::extents<numRows_A, numCols_A>, Layout_A, Accessor_A> A,
  Triangle t,
  DiagonalStorage d,
  Side s,
  std::experimental::basic_mdspan<ElementType_B, std::experimental::extents<numRows_B, numCols_B>, Layout_B, Accessor_B> B,
  std::experimental::basic_mdspan<ElementType_X, std::experimental::extents<numRows_X, numCols_X>, Layout_X, Accessor_X> X)
{
  triangular_matrix_matrix_solve(A, t, d, s, B, X);
}

} // end namespace linalg
} // end inline namespace __p1673_version_0
} // end namespace experimental
} // end namespace std

#endif //LINALG_INCLUDE_EXPERIMENTAL___P1673_BITS_BLAS3_TRIANGULAR_MATRIX_MATRIX_SOLVE_HPP_
