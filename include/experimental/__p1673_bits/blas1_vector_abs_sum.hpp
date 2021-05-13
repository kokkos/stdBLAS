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

#ifndef LINALG_INCLUDE_EXPERIMENTAL___P1673_BITS_BLAS1_VECTOR_ABS_SUM_HPP_
#define LINALG_INCLUDE_EXPERIMENTAL___P1673_BITS_BLAS1_VECTOR_ABS_SUM_HPP_

#include <cstdlib>
#include <cmath>

namespace std {
namespace experimental {
inline namespace __p1673_version_0 {

template<class in_vector_t,
         class Scalar>
void vector_abs_sum(in_vector_t v,
                    Scalar& result)
{
  using std::abs;

  result = 0.0;
  ptrdiff_t n = v.extent(0);
  if (n == 0) {
    return;
  }
  else if (n == ptrdiff_t(1)) {
    result = abs(v(0));
    return;
  }

  // Clean-up loop for sizes that aren't evenly divisible by 6
  ptrdiff_t m = n%6;
  if (m != 0) {
    for (ptrdiff_t i = 0; i < m; ++i) {
      result += abs(v(i));
    }
    if (n < 6) {
      return;
    }
  }
  // Explicit loop unrolling
  for (ptrdiff_t i = m; i < n; i+=6) {
    result += abs(v(i))   + abs(v(i+1)) + abs(v(i+2)) +
              abs(v(i+3)) + abs(v(i+4)) + abs(v(i+5));
  }
}

template<class ExecutionPolicy,
         class in_vector_t,
         class Scalar>
void vector_abs_sum(ExecutionPolicy&& /* exec */,
                    in_vector_t v,
                    Scalar& result)
{
  vector_abs_sum(v, result);
}

} // end inline namespace __p1673_version_0
} // end namespace experimental
} // end namespace std

#endif //LINALG_INCLUDE_EXPERIMENTAL___P1673_BITS_BLAS1_VECTOR_ABS_SUM_HPP_
