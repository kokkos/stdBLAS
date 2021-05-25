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

#ifndef LINALG_INCLUDE_EXPERIMENTAL___P1673_BITS_BLAS1_VECTOR_IDX_ABS_MAX_HPP_
#define LINALG_INCLUDE_EXPERIMENTAL___P1673_BITS_BLAS1_VECTOR_IDX_ABS_MAX_HPP_

namespace std {
namespace experimental {
inline namespace __p1673_version_0 {
namespace linalg {

template<class in_vector_t>
ptrdiff_t vector_idx_abs_max(in_vector_t v)
{
  using std::abs;
  using element_type = typename in_vector_t::element_type;
  using magnitude_type = decltype(abs(v(0)));

  if (v.extent(0) == 0) {
    return -1;
  }
  ptrdiff_t maxInd = 0;
  magnitude_type maxVal = abs(v(0));
  for (ptrdiff_t i = 1; i < v.extent(0); ++i) {
    if (maxVal < abs(v(i))) {
      maxVal = abs(v(i));
      maxInd = i;
    }
  }
  return maxInd; // FIXME check for NaN "never less than" stuff
}


template<class ExecutionPolicy,
         class in_vector_t>
ptrdiff_t vector_idx_abs_max(ExecutionPolicy&& /* exec */,
                             in_vector_t v)
{
  return vector_idx_abs_max(v);
}

} // end namespace linalg
} // end inline namespace __p1673_version_0
} // end namespace experimental
} // end namespace std

#endif //LINALG_INCLUDE_EXPERIMENTAL___P1673_BITS_BLAS1_VECTOR_IDX_ABS_MAX_HPP_
