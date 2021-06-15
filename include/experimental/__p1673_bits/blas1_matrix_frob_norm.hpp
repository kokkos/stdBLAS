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

#ifndef LINALG_INCLUDE_EXPERIMENTAL___P1673_BITS_BLAS1_MATRIX_FROB_NORM_HPP_
#define LINALG_INCLUDE_EXPERIMENTAL___P1673_BITS_BLAS1_MATRIX_FROB_NORM_HPP_

#include <cmath>
#include <cstdlib>

namespace std {
namespace experimental {
inline namespace __p1673_version_0 {
namespace linalg {

template<
    class ElementType,
    extents<>::size_type numRows, 
    extents<>::size_type numCols,
    class Layout,
    class Accessor,
    class Scalar>
Scalar matrix_frob_norm(
  std::experimental::basic_mdspan<ElementType, std::experimental::extents<numRows, numCols>, Layout, Accessor> A,
  Scalar init)
{
  using std::abs;
  using std::sqrt;
  using size_type = typename extents<>::size_type;

  // Handle special cases.
  auto result = init;
  if (A.extent(0) == 0 || A.extent(1) == 0) {
    return result;
  }
  else if(A.extent(0) == size_type(1) && A.extent(1) == size_type(1)) {
    result += abs(A(0, 0));
    return result;
  }

  // Rescaling avoids unwarranted overflow or underflow.
  Scalar scale = 0.0;
  Scalar ssq = 1.0;
  for (size_type i = 0; i < A.extent(0); ++i) {
    for (size_type j = 0; j < A.extent(1); ++j) {
      const auto absaij = abs(A(i,j));
      if (absaij != 0.0) {
        const auto quotient = scale / absaij;
        if (scale < absaij) {
          ssq = Scalar(1.0) + ssq * quotient * quotient;
          scale = absaij;
        }
        else {
          ssq = ssq + quotient * quotient;
        }
      }
    }
  }
  result += scale * sqrt(ssq);
  return result;
}

template<class ExecutionPolicy,
  class ElementType,
  extents<>::size_type numRows, 
  extents<>::size_type numCols,
  class Layout,
  class Accessor,
  class Scalar>
Scalar matrix_frob_norm(
  ExecutionPolicy&& /* exec */,
  std::experimental::basic_mdspan<ElementType, std::experimental::extents<numRows, numCols>, Layout, Accessor> A,
  Scalar init)
{
  return matrix_frob_norm(A, init);
}

// TODO: Implement auto functions
#if 0
template<class in_matrix_t>
auto matrix_frob_norm(in_matrix_t A)
{
 
}

template<class ExecutionPolicy,
         class in_matrix_t>
auto matrix_frob_norm(ExecutionPolicy&& exec,
                      in_matrix_t A)
{

}
#endif

} // end namespace linalg
} // end inline namespace __p1673_version_0
} // end namespace experimental
} // end namespace std

#endif //LINALG_INCLUDE_EXPERIMENTAL___P1673_BITS_BLAS1_MATRIX_FROB_NORM_HPP_
