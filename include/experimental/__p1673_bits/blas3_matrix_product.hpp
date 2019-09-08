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

#ifndef LINALG_INCLUDE_EXPERIMENTAL___P1673_BITS_BLAS3_MATRIX_PRODUCT_HPP_
#define LINALG_INCLUDE_EXPERIMENTAL___P1673_BITS_BLAS3_MATRIX_PRODUCT_HPP_

namespace std {
namespace experimental {
inline namespace __p1673_version_0 {

// Overwriting general matrix-matrix product

template<class in_matrix_1_t,
         class in_matrix_2_t,
         class out_matrix_t>
void matrix_product(in_matrix_1_t A,
                    in_matrix_2_t B,
                    out_matrix_t C)
{
  for(ptrdiff_t i = 0; i < C.extent(0); ++i) {
    for(ptrdiff_t j = 0; j < C.extent(1); ++j) {
      C(i,j) = 0.0;
      for(ptrdiff_t k = 0; k < A.extent(1); ++k) {
        C(i,j) += A(i,k) * B(k,j);
      }
    }
  }
}

template<class ExecutionPolicy,
         class in_matrix_1_t,
         class in_matrix_2_t,
         class out_matrix_t>
void matrix_product(ExecutionPolicy&& /* exec */,
                    in_matrix_1_t A,
                    in_matrix_2_t B,
                    out_matrix_t C)
{
  matrix_product(A, B, C);
}

// Updating general matrix-matrix product

template<class in_matrix_1_t,
         class in_matrix_2_t,
         class in_matrix_3_t,
         class out_matrix_t>
void matrix_product(in_matrix_1_t A,
                    in_matrix_2_t B,
                    in_matrix_3_t E,
                    out_matrix_t C)
{
  for(ptrdiff_t i = 0; i < C.extent(0); ++i) {
    for(ptrdiff_t j = 0; j < C.extent(1); ++j) {
      C(i,j) = 0.0;
      for(ptrdiff_t k = 0; k < A.extent(1); ++k) {
        C(i,j) += E(i,j) + A(i,k) * B(k,j);
      }
    }
  }
}

template<class ExecutionPolicy,
         class in_matrix_1_t,
         class in_matrix_2_t,
         class in_matrix_3_t,
         class out_matrix_t>
void matrix_product(ExecutionPolicy&& /* exec */,
                    in_matrix_1_t A,
                    in_matrix_2_t B,
                    in_matrix_3_t E,
                    out_matrix_t C)
{
  matrix_product(A, B, E, C);
}

// Overwriting symmetric matrix-matrix product

template<class in_matrix_1_t,
         class Triangle,
         class Side,
         class in_matrix_2_t,
         class out_matrix_t>
void symmetric_matrix_product(
  in_matrix_1_t A,
  Triangle t,
  Side s,
  in_matrix_2_t B,
  out_matrix_t C)
{
  if constexpr (std::is_same_v<Side, left_side_t>) {
    if constexpr (std::is_same_v<Triangle, lower_triangle_t>) {
      for (ptrdiff_t j = 0; j < C.extent(1); ++j) {
        for (ptrdiff_t i = j; i < C.extent(0); ++i) {
          C(i,j) = 0.0;
          for (ptrdiff_t k = 0; k < A.extent(1); ++k) {
            C(i,j) += A(i,k) * B(k,j);
          }
        }
      }
    }
    else { // upper_triangle_t
      for (ptrdiff_t j = 0; j < C.extent(1); ++j) {      
        for (ptrdiff_t i = 0; i <= j; ++i) {
          C(i,j) = 0.0;
          for (ptrdiff_t k = 0; k < A.extent(1); ++k) {
            C(i,j) += A(i,k) * B(k,j);
          }
        }
      }
    }
  }
  else { // right_side_t
    if constexpr (std::is_same_v<Triangle, lower_triangle_t>) {
      for (ptrdiff_t j = 0; j < C.extent(1); ++j) {
        for (ptrdiff_t i = j; i < C.extent(0); ++i) {
          C(i,j) = 0.0;
          for (ptrdiff_t k = 0; k < A.extent(1); ++k) {
            C(i,j) += B(i,k) * A(k,j);
          }
        }
      }
    }
    else { // upper_triangle_t
      for (ptrdiff_t j = 0; j < C.extent(1); ++j) {      
        for (ptrdiff_t i = 0; i <= j; ++i) {
          C(i,j) = 0.0;
          for (ptrdiff_t k = 0; k < A.extent(1); ++k) {
            C(i,j) += B(i,k) * A(k,j);
          }
        }
      }
    }
  }
}

template<class ExecutionPolicy,
         class in_matrix_1_t,
         class Triangle,
         class Side,
         class in_matrix_2_t,
         class out_matrix_t>
void symmetric_matrix_product(
  ExecutionPolicy&& /* exec */,
  in_matrix_1_t A,
  Triangle t,
  Side s,
  in_matrix_2_t B,
  out_matrix_t C)
{
  symmetric_matrix_product(A, t, s, B, C);
}

// Updating symmetric matrix-matrix product

template<class in_matrix_1_t,
         class Triangle,
         class Side,
         class in_matrix_2_t,
         class in_matrix_3_t,
         class out_matrix_t>
void symmetric_matrix_product(
  in_matrix_1_t A,
  Triangle t,
  Side s,
  in_matrix_2_t B,
  in_matrix_3_t E,
  out_matrix_t C)
{
  if constexpr (std::is_same_v<Side, left_side_t>) {
    if constexpr (std::is_same_v<Triangle, lower_triangle_t>) {
      for (ptrdiff_t j = 0; j < C.extent(1); ++j) {
        for (ptrdiff_t i = j; i < C.extent(0); ++i) {
          C(i,j) = 0.0;
          for (ptrdiff_t k = 0; k < A.extent(1); ++k) {
            C(i,j) += E(i,j) + A(i,k) * B(k,j);
          }
        }
      }
    }
    else { // upper_triangle_t
      for (ptrdiff_t j = 0; j < C.extent(1); ++j) {      
        for (ptrdiff_t i = 0; i <= j; ++i) {
          C(i,j) = 0.0;
          for (ptrdiff_t k = 0; k < A.extent(1); ++k) {
            C(i,j) += E(i,j) + A(i,k) * B(k,j);
          }
        }
      }
    }
  }
  else { // right_side_t
    if constexpr (std::is_same_v<Triangle, lower_triangle_t>) {
      for (ptrdiff_t j = 0; j < C.extent(1); ++j) {
        for (ptrdiff_t i = j; i < C.extent(0); ++i) {
          C(i,j) = 0.0;
          for (ptrdiff_t k = 0; k < A.extent(1); ++k) {
            C(i,j) += E(i,j) + B(i,k) * A(k,j);
          }
        }
      }
    }
    else { // upper_triangle_t
      for (ptrdiff_t j = 0; j < C.extent(1); ++j) {      
        for (ptrdiff_t i = 0; i <= j; ++i) {
          C(i,j) = 0.0;
          for (ptrdiff_t k = 0; k < A.extent(1); ++k) {
            C(i,j) += E(i,j) + B(i,k) * A(k,j);
          }
        }
      }
    }
  }
}

template<class ExecutionPolicy,
         class in_matrix_1_t,
         class Triangle,
         class Side,
         class in_matrix_2_t,
         class in_matrix_3_t,
         class out_matrix_t>
void symmetric_matrix_product(
  ExecutionPolicy&& /* exec */,
  in_matrix_1_t A,
  Triangle t,
  Side s,
  in_matrix_2_t B,
  in_matrix_3_t E,
  out_matrix_t C)
{
  symmetric_matrix_product(A, t, s, B, E, C);
}

// Overwriting Hermitian matrix-matrix product

template<class in_matrix_1_t,
         class Triangle,
         class Side,
         class in_matrix_2_t,
         class out_matrix_t>
void hermitian_matrix_product(
  in_matrix_1_t A,
  Triangle t,
  Side s,
  in_matrix_2_t B,
  out_matrix_t C);

template<class ExecutionPolicy,
         class in_matrix_1_t,
         class Triangle,
         class Side,
         class in_matrix_2_t,
         class out_matrix_t>
void hermitian_matrix_product(
  ExecutionPolicy&& exec,
  in_matrix_1_t A,
  Triangle t,
  Side s,
  in_matrix_2_t B,
  out_matrix_t C);

// Updating Hermitian matrix-matrix product

template<class in_matrix_1_t,
         class Triangle,
         class Side,
         class in_matrix_2_t,
         class in_matrix_3_t,
         class out_matrix_t>
void hermitian_matrix_product(
  in_matrix_1_t A,
  Triangle t,
  Side s,
  in_matrix_2_t B,
  in_matrix_3_t E,
  out_matrix_t C);

template<class ExecutionPolicy,
         class in_matrix_1_t,
         class Triangle,
         class Side,
         class in_matrix_2_t,
         class in_matrix_3_t,
         class out_matrix_t>
void hermitian_matrix_product(
  ExecutionPolicy&& exec,
  in_matrix_1_t A,
  Triangle t,
  Side s,
  in_matrix_2_t B,
  in_matrix_3_t E,
  out_matrix_t C);

} // end inline namespace __p1673_version_0
} // end namespace experimental
} // end namespace std

#endif //LINALG_INCLUDE_EXPERIMENTAL___P1673_BITS_BLAS3_MATRIX_PRODUCT_HPP_
