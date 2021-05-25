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

#ifndef LINALG_INCLUDE_EXPERIMENTAL___P1673_BITS_BLAS2_MATRIX_VECTOR_SOLVE_HPP_
#define LINALG_INCLUDE_EXPERIMENTAL___P1673_BITS_BLAS2_MATRIX_VECTOR_SOLVE_HPP_

#include <type_traits>

namespace std {
namespace experimental {
inline namespace __p1673_version_0 {
namespace linalg {

namespace {

template<class in_matrix_t,
         class DiagonalStorage,
         class in_vector_t,
         class out_vector_t>
void trsv_upper_triangular_left_side(
  in_matrix_t A,
  DiagonalStorage d,
  in_vector_t B,
  out_vector_t X)
{
  constexpr bool explicit_diagonal =
    std::is_same_v<DiagonalStorage, explicit_diagonal_t>;

  const ptrdiff_t A_num_rows = A.extent(0);
  const ptrdiff_t A_num_cols = A.extent(1);

  // One advantage of using signed index types is that you can write
  // descending loops with zero-based indices.
  for (ptrdiff_t i = A_num_rows - 1; i >= 0; --i) {
    // TODO this would be a great opportunity for an implementer to
    // add value, by accumulating in extended precision (or at least
    // in a type with the max precision of X and B).
    using sum_type = decltype (B(i) - A(0,0) * X(0));
    //using sum_type = typename out_object_t::element_type;
    const sum_type t (B(i));
    for (ptrdiff_t j = i + 1; j < A_num_cols; ++j) {
      t = t - A(i,j) * X(j);
    }
    if constexpr (explicit_diagonal) {
      X(i) = t / A(i,i);
    }
    else {
      X(i) = t;
    }
  }
}

template<class in_matrix_t,
         class DiagonalStorage,
         class in_vector_t,
         class out_vector_t>
void trsv_lower_triangular_left_side(
  in_matrix_t A,
  DiagonalStorage d,
  in_vector_t B,
  out_vector_t X)
{
  constexpr bool explicit_diagonal =
    std::is_same_v<DiagonalStorage, explicit_diagonal_t>;

  const ptrdiff_t A_num_rows = A.extent(0);
  const ptrdiff_t A_num_cols = A.extent(1);

  for (ptrdiff_t i = 0; i < A_num_rows; ++i) {
    // TODO this would be a great opportunity for an implementer to
    // add value, by accumulating in extended precision (or at least
    // in a type with the max precision of X and B).
    using sum_type = decltype (B(i) - A(0,0) * X(0));
    //using sum_type = typename out_object_t::element_type;
    const sum_type t (B(i));
    for (ptrdiff_t j = 0; j < i; ++j) {
      t = t - A(i,j) * X(j);
    }
    if constexpr (explicit_diagonal) {
      X(i) = t / A(i,i);
    }
    else {
      X(i) = t;
    }
  }
}

}

template<class in_matrix_t,
         class Triangle,
         class DiagonalStorage,
         class in_vector_t,
         class out_vector_t>
void triangular_matrix_vector_solve(
  in_matrix_t A,
  Triangle t,
  DiagonalStorage d,
  in_vector_t b,
  out_vector_t x)
{
  if constexpr (std::is_same_v<Triangle, lower_triangle_t>) {
    trsv_lower_triangular_left_side(A, d, b, x);
  }
  else {
    trsv_upper_triangular_left_side(A, d, b, x);
  }
}

template<class ExecutionPolicy,
         class in_matrix_t,
         class Triangle,
         class DiagonalStorage,
         class in_vector_t,
         class out_vector_t>
void triangular_matrix_vector_solve(
  ExecutionPolicy&& /* exec */,
  in_matrix_t A,
  Triangle t,
  DiagonalStorage d,
  in_vector_t b,
  out_vector_t x)
{
  triangular_matrix_vector_solve(A, t, d, b, x);
}

} // end namespace linalg
} // end inline namespace __p1673_version_0
} // end namespace experimental
} // end namespace std

#endif //LINALG_INCLUDE_EXPERIMENTAL___P1673_BITS_BLAS2_MATRIX_VECTOR_SOLVE_HPP_
