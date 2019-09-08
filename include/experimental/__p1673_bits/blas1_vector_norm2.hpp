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

#ifndef LINALG_INCLUDE_EXPERIMENTAL___P1673_BITS_BLAS1_VECTOR_NORM2_HPP_
#define LINALG_INCLUDE_EXPERIMENTAL___P1673_BITS_BLAS1_VECTOR_NORM2_HPP_

#include <cmath>
#include <cstdlib>

namespace std {
namespace experimental {
inline namespace __p1673_version_0 {

template<class in_vector_t,
         class Scalar>
void vector_norm2(in_vector_t x,
                  Scalar& result)
{
  using std::abs;
  using std::sqrt;

  // Rescaling, as in the Reference BLAS DNRM2 implementation, avoids
  // unwarranted overflow or underflow.

  result = 0.0;
  if (x.extent(0) == 0) {
    return;
  }
  else if (x.extent(0) == ptrdiff_t(1)) {
    result = abs(x(0));
    return;
  }

  Scalar scale = 0.0;
  Scalar ssq = 1.0;
  for (ptrdiff_t i = 0; i < x.extent(0); ++i) {
    if (abs(x(i)) != 0.0) {
      const auto absxi = abs(x(i));
      const auto quotient = scale / absxi;
      if (scale < absxi) {
        ssq = Scalar(1.0) + ssq * quotient * quotient;
        scale = absxi;
      }
      else {
        ssq = ssq + quotient * quotient;
      }
    }
  }
  result = scale * sqrt(ssq);
}

template<class ExecutionPolicy,
         class in_vector_t,
         class Scalar>
void vector_norm2(ExecutionPolicy&& /* exec */,
                  in_vector_t v,
                  Scalar& result)
{
  vector_norm2(v, result);
}

} // end inline namespace __p1673_version_0
} // end namespace experimental
} // end namespace std

#endif //LINALG_INCLUDE_EXPERIMENTAL___P1673_BITS_BLAS1_VECTOR_NORM2_HPP_
