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

namespace {

template<class in_matrix_t,
         class DiagonalStorage,
         class in_object_t,
         class out_object_t>
void trsm_upper_triangular_left_side(
  in_matrix_t A,
  DiagonalStorage d,
  in_object_t B,
  out_object_t X)
{
  constexpr bool explicit_diagonal =
    std::is_same_v<DiagonalStorage, explicit_diagonal_t>;

  const ptrdiff_t A_num_rows = A.extent(0);
  const ptrdiff_t B_num_cols = B.extent(1);

  for (ptrdiff_t k = 0; k < B_num_cols; ++k) {
    // One advantage of using signed index types is that you can write
    // descending loops with zero-based indices.
    for (ptrdiff_t i = A_num_rows - 1; i >= 0; --i) {
      // TODO this would be a great opportunity for an implementer to
      // add value, by accumulating in extended precision (or at least
      // in a type with the max precision of X and B).
      using sum_type = decltype (B(i,k) - A(0,0) * X(0,0));
      //using sum_type = typename out_object_t::element_type;
      const sum_type t (B(i,k));
      for (ptrdiff_t j = i + 1; j < A_num_rows; ++j) {
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

template<class in_matrix_t,
         class DiagonalStorage,
         class in_object_t,
         class out_object_t>
void trsm_lower_triangular_left_side(
  in_matrix_t A,
  DiagonalStorage d,
  in_object_t B,
  out_object_t X)
{
  constexpr bool explicit_diagonal =
    std::is_same_v<DiagonalStorage, explicit_diagonal_t>;

  const ptrdiff_t A_num_rows = A.extent(0);
  const ptrdiff_t B_num_cols = B.extent(1);

  for (ptrdiff_t k = 0; k < B_num_cols; ++k) {
    for (ptrdiff_t i = 0; i < A_num_rows; ++i) {
      // TODO this would be a great opportunity for an implementer to
      // add value, by accumulating in extended precision (or at least
      // in a type with the max precision of X and B).
      using x_element_type = typename out_object_t::element_type;
      const x_element_type t (B(i,k));
      for (ptrdiff_t j = 0; j < i; ++j) {
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

template<class in_matrix_t,
         class DiagonalStorage,
         class in_object_t,
         class out_object_t>
void trsm_upper_triangular_right_side(
  in_matrix_t A,
  DiagonalStorage d,
  in_object_t B,
  out_object_t X)
{
  constexpr bool explicit_diagonal =
    std::is_same_v<DiagonalStorage, explicit_diagonal_t>;

  const ptrdiff_t B_num_rows = B.extent(0);
  const ptrdiff_t A_num_cols = A.extent(1);

  for (ptrdiff_t i = 0; i < B_num_rows; ++i) {
    for (ptrdiff_t j = 0; j < A_num_cols; ++j) {
      using sum_type = decltype (B(i,j) - A(0,0) * X(0,0));
      const sum_type t (B(i,j));
      for (ptrdiff_t k = 0; k < j; ++j) {
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

template<class in_matrix_t,
         class DiagonalStorage,
         class in_object_t,
         class out_object_t>
void trsm_lower_triangular_right_side(
  in_matrix_t A,
  DiagonalStorage d,
  in_object_t B,
  out_object_t X)
{
  constexpr bool explicit_diagonal =
    std::is_same_v<DiagonalStorage, explicit_diagonal_t>;

  const ptrdiff_t B_num_rows = B.extent(0);
  const ptrdiff_t A_num_rows = A.extent(0);
  const ptrdiff_t A_num_cols = A.extent(1);

  for (ptrdiff_t i = 0; i < B_num_rows; ++i) {
    for (ptrdiff_t j = 0; j < A_num_cols; ++j) {
      using sum_type = decltype (B(i,j) - A(0,0) * X(0,0));
      const sum_type t (B(i,j));
      for (ptrdiff_t k = j + 1; k < A_num_rows; ++j) {
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

}

template<class in_matrix_t,
         class Triangle,
         class DiagonalStorage,
         class Side,
         class in_object_t,
         class out_object_t>
void triangular_matrix_matrix_solve(
  in_matrix_t A,
  Triangle t,
  DiagonalStorage d,
  Side s,
  in_object_t B,
  out_object_t X)
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
         class in_matrix_t,
         class Triangle,
         class DiagonalStorage,
         class Side,
         class in_object_t,
         class out_object_t>
void triangular_matrix_matrix_solve(
  ExecutionPolicy&& /* exec */,
  in_matrix_t A,
  Triangle t,
  DiagonalStorage d,
  Side s,
  in_object_t B,
  out_object_t X)
{
  triangular_matrix_matrix_solve(A, t, d, s, B, X);
}

} // end inline namespace __p1673_version_0
} // end namespace experimental
} // end namespace std

#endif //LINALG_INCLUDE_EXPERIMENTAL___P1673_BITS_BLAS3_TRIANGULAR_MATRIX_MATRIX_SOLVE_HPP_
