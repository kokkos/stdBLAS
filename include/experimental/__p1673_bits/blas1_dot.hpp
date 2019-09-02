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

#ifndef LINALG_INCLUDE_EXPERIMENTAL___P1673_BITS_BLAS1_DOT_HPP_
#define LINALG_INCLUDE_EXPERIMENTAL___P1673_BITS_BLAS1_DOT_HPP_

#include <type_traits>

namespace std {
namespace experimental {
inline namespace __p1673_version_0 {

// Non-conjugated dot         
template<class in_vector_1_t,
         class in_vector_2_t,
         class Scalar>
void dot(in_vector_1_t v1,
         in_vector_2_t v2,
         Scalar& result)
{
  Scalar myResult {};
  for (size_t k = 0; k < v1.extent(0); ++k) {
    myResult += v1(k) * v2(k);
  }
  result = myResult;
}

template<class ExecutionPolicy,
         class in_vector_1_t,
         class in_vector_2_t,
         class Scalar>
void dot(ExecutionPolicy&& /* exec */,
         in_vector_1_t v1,
         in_vector_2_t v2,
         Scalar& result)
{
  dot (v1, v2, result);
}

// Conjugated dot

template<class in_vector_1_t,
         class in_vector_2_t,
         class Scalar>
void dotc(in_vector_1_t v1,
          in_vector_2_t v2,
          Scalar& result)
{
  // FIXME We don't have an mdarray implementation of conjugate_view
  // yet.  See #10.
  //
  // FIXME This compiles, but doesn't actually work; it returns zero.
  
  //dot(v1, conjugate_view(v2), result);

  Scalar myResult {};
  for (size_t k = 0; k < v1.extent(0); ++k) {
    if constexpr (std::is_same_v<Scalar, std::complex<float>> ||
                  std::is_same_v<Scalar, std::complex<double>> ||
                  std::is_same_v<Scalar, std::complex<long double>>) {
      using std::conj;
      myResult += v1(k) * conj(v2(k));
    }
    else {
      myResult += v1(k) * v2(k);
    }
  }
  result = myResult;
}

template<class ExecutionPolicy,
         class in_vector_1_t,
         class in_vector_2_t,
         class Scalar>
void dotc(ExecutionPolicy&& /* exec */,
          in_vector_1_t v1,
          in_vector_2_t v2,
          Scalar& result)
{
  dotc(v1, v2, result);
}
         
} // end inline namespace __p1673_version_0
} // end namespace experimental
} // end namespace std

#endif //LINALG_INCLUDE_EXPERIMENTAL___P1673_BITS_BLAS1_DOT_HPP_
