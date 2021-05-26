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

template<class in_matrix_1_t,
         class in_matrix_2_t,
         class inout_matrix_t,
         class Triangle>
void symmetric_matrix_rank_2k_update(
  in_matrix_1_t A,
  in_matrix_2_t B,
  inout_matrix_t C,
  Triangle /* t */)
{
  constexpr bool lower_tri =
    std::is_same_v<Triangle, lower_triangle_t>;
  for (ptrdiff_t j = 0; j < C.extent(1); ++j) {
    const ptrdiff_t i_lower = lower_tri ? j : ptrdiff_t(0);
    const ptrdiff_t i_upper = lower_tri ? C.extent(0) : j+1;
    for (ptrdiff_t i = i_lower; i < i_upper; ++i) {
      for (ptrdiff_t k = 0; k < A.extent(1); ++k) {
        C(i,j) += A(i,k)*B(j,k) + B(i,k)*A(j,k);
      }
    }
  }
}

template<class ExecutionPolicy,
         class in_matrix_1_t,
         class in_matrix_2_t,
         class inout_matrix_t,
         class Triangle>
void symmetric_matrix_rank_2k_update(
  ExecutionPolicy&& /* exec */,
  in_matrix_1_t A,
  in_matrix_2_t B,
  inout_matrix_t C,
  Triangle t)
{
  symmetric_matrix_rank_2k_update(A, B, C, t);
}

// Rank-2k update of a Hermitian matrix

template<class in_matrix_1_t,
         class in_matrix_2_t,
         class inout_matrix_t,
         class Triangle>
void hermitian_matrix_rank_2k_update(
  in_matrix_1_t A,
  in_matrix_2_t B,
  inout_matrix_t C,
  Triangle /* t */)
{
  using std::conj;
  constexpr bool lower_tri =
    std::is_same_v<Triangle, lower_triangle_t>;
  for (ptrdiff_t j = 0; j < C.extent(1); ++j) {
    const ptrdiff_t i_lower = lower_tri ? j : ptrdiff_t(0);
    const ptrdiff_t i_upper = lower_tri ? C.extent(0) : j+1;
    for (ptrdiff_t i = i_lower; i < i_upper; ++i) {
      for (ptrdiff_t k = 0; k < A.extent(1); ++k) {
        C(i,j) += A(i,k) * conj(B(j,k)) + B(i,k) * conj(A(j,k));
      }
    }
  }
}

template<class ExecutionPolicy,
         class in_matrix_1_t,
         class in_matrix_2_t,
         class inout_matrix_t,
         class Triangle>
void hermitian_matrix_rank_2k_update(
  ExecutionPolicy&& /* exec */,
  in_matrix_1_t A,
  in_matrix_2_t B,
  inout_matrix_t C,
  Triangle t)
{
  hermitian_matrix_rank_2k_update(A, B, C, t);
}

} // end namespace linalg
} // end inline namespace __p1673_version_0
} // end namespace experimental
} // end namespace std

#endif //LINALG_INCLUDE_EXPERIMENTAL___P1673_BITS_BLAS3_MATRIX_RANK_2K_UPDATE_HPP_
