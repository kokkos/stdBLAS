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

#ifndef LINALG_INCLUDE_EXPERIMENTAL___P1673_BITS_BLAS3_MATRIX_RANK_2K_UPDATE_HPP_
#define LINALG_INCLUDE_EXPERIMENTAL___P1673_BITS_BLAS3_MATRIX_RANK_2K_UPDATE_HPP_

namespace std {
namespace experimental {
inline namespace __p1673_version_0 {
namespace linalg {

// Rank-2k update of a symmetric matrix

template<class ElementType_A,
         extents<>::size_type numRows_A,
         extents<>::size_type numCols_A,
         class Layout_A,
         class Accessor_A,
         class ElementType_B,
         extents<>::size_type numRows_B,
         extents<>::size_type numCols_B,
         class Layout_B,
         class Accessor_B,
         class ElementType_C,
         extents<>::size_type numRows_C,
         extents<>::size_type numCols_C,
         class Layout_C,
         class Accessor_C,
         class Triangle>
void symmetric_matrix_rank_2k_update(
  std::experimental::basic_mdspan<ElementType_A, std::experimental::extents<numRows_A, numCols_A>, Layout_A, Accessor_A> A,
  std::experimental::basic_mdspan<ElementType_B, std::experimental::extents<numRows_B, numCols_B>, Layout_B, Accessor_B> B,
  std::experimental::basic_mdspan<ElementType_C, std::experimental::extents<numRows_C, numCols_C>, Layout_C, Accessor_C> C,
  Triangle /* t */)
{
  constexpr bool lower_tri =
    std::is_same_v<Triangle, lower_triangle_t>;
  using size_type = typename extents<>::size_type;

  for (size_type j = 0; j < C.extent(1); ++j) {
    const size_type i_lower = lower_tri ? j : size_type(0);
    const size_type i_upper = lower_tri ? C.extent(0) : j+1;
    for (size_type i = i_lower; i < i_upper; ++i) {
      for (size_type k = 0; k < A.extent(1); ++k) {
        C(i,j) += A(i,k)*B(j,k) + B(i,k)*A(j,k);
      }
    }
  }
}

template<class ExecutionPolicy,
         class ElementType_A,
         extents<>::size_type numRows_A,
         extents<>::size_type numCols_A,
         class Layout_A,
         class Accessor_A,
         class ElementType_B,
         extents<>::size_type numRows_B,
         extents<>::size_type numCols_B,
         class Layout_B,
         class Accessor_B,
         class ElementType_C,
         extents<>::size_type numRows_C,
         extents<>::size_type numCols_C,
         class Layout_C,
         class Accessor_C,
         class Triangle>
void symmetric_matrix_rank_2k_update(
  ExecutionPolicy&& /* exec */,
  std::experimental::basic_mdspan<ElementType_A, std::experimental::extents<numRows_A, numCols_A>, Layout_A, Accessor_A> A,
  std::experimental::basic_mdspan<ElementType_B, std::experimental::extents<numRows_B, numCols_B>, Layout_B, Accessor_B> B,
  std::experimental::basic_mdspan<ElementType_C, std::experimental::extents<numRows_C, numCols_C>, Layout_C, Accessor_C> C,
  Triangle t)
{
  symmetric_matrix_rank_2k_update(A, B, C, t);
}

// Rank-2k update of a Hermitian matrix

template<class ElementType_A,
         extents<>::size_type numRows_A,
         extents<>::size_type numCols_A,
         class Layout_A,
         class Accessor_A,
         class ElementType_B,
         extents<>::size_type numRows_B,
         extents<>::size_type numCols_B,
         class Layout_B,
         class Accessor_B,
         class ElementType_C,
         extents<>::size_type numRows_C,
         extents<>::size_type numCols_C,
         class Layout_C,
         class Accessor_C,
         class Triangle>
void hermitian_matrix_rank_2k_update(
  std::experimental::basic_mdspan<ElementType_A, std::experimental::extents<numRows_A, numCols_A>, Layout_A, Accessor_A> A,
  std::experimental::basic_mdspan<ElementType_B, std::experimental::extents<numRows_B, numCols_B>, Layout_B, Accessor_B> B,
  std::experimental::basic_mdspan<ElementType_C, std::experimental::extents<numRows_C, numCols_C>, Layout_C, Accessor_C> C,
  Triangle /* t */)
{
  using std::conj;
  constexpr bool lower_tri =
    std::is_same_v<Triangle, lower_triangle_t>;
  using size_type = typename extents<>::size_type;

  for (size_type j = 0; j < C.extent(1); ++j) {
    const size_type i_lower = lower_tri ? j : size_type(0);
    const size_type i_upper = lower_tri ? C.extent(0) : j+1;
    for (size_type i = i_lower; i < i_upper; ++i) {
      for (size_type k = 0; k < A.extent(1); ++k) {
        C(i,j) += A(i,k) * conj(B(j,k)) + B(i,k) * conj(A(j,k));
      }
    }
  }
}

template<class ExecutionPolicy,
         class ElementType_A,
         extents<>::size_type numRows_A,
         extents<>::size_type numCols_A,
         class Layout_A,
         class Accessor_A,
         class ElementType_B,
         extents<>::size_type numRows_B,
         extents<>::size_type numCols_B,
         class Layout_B,
         class Accessor_B,
         class ElementType_C,
         extents<>::size_type numRows_C,
         extents<>::size_type numCols_C,
         class Layout_C,
         class Accessor_C,
         class Triangle>
void hermitian_matrix_rank_2k_update(
  ExecutionPolicy&& /* exec */,
  std::experimental::basic_mdspan<ElementType_A, std::experimental::extents<numRows_A, numCols_A>, Layout_A, Accessor_A> A,
  std::experimental::basic_mdspan<ElementType_B, std::experimental::extents<numRows_B, numCols_B>, Layout_B, Accessor_B> B,
  std::experimental::basic_mdspan<ElementType_C, std::experimental::extents<numRows_C, numCols_C>, Layout_C, Accessor_C> C,
  Triangle t)
{
  hermitian_matrix_rank_2k_update(A, B, C, t);
}

} // end namespace linalg
} // end inline namespace __p1673_version_0
} // end namespace experimental
} // end namespace std

#endif //LINALG_INCLUDE_EXPERIMENTAL___P1673_BITS_BLAS3_MATRIX_RANK_2K_UPDATE_HPP_
