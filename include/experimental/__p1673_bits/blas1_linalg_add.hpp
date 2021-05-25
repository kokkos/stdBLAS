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

#ifndef LINALG_INCLUDE_EXPERIMENTAL___P1673_BITS_BLAS1_LINALG_ADD_HPP_
#define LINALG_INCLUDE_EXPERIMENTAL___P1673_BITS_BLAS1_LINALG_ADD_HPP_

namespace std {
namespace experimental {
inline namespace __p1673_version_0 {
namespace linalg {

namespace {

template<class in_vector_1_t,
         class in_vector_2_t,
         class out_vector_t>
void add_rank_1(in_vector_1_t x,
                in_vector_2_t y,
                out_vector_t z)
{
  for (ptrdiff_t i = 0; i < z.extent(0); ++i) {
    z(i) = x(i) + y(i);
  }
}

template<class in_matrix_1_t,
         class in_matrix_2_t,
         class out_matrix_t>
void add_rank_2(in_matrix_1_t x,
                in_matrix_2_t y,
                out_matrix_t z)
{
  for (ptrdiff_t j = 0; j < x.extent(1); ++j) {
    for (ptrdiff_t i = 0; i < x.extent(0); ++i) {
      z(i,j) = x(i,j) + y(i,j);
    }
  }
}

// TODO add mdarray specializations; needed so that out_*_t is
// not passed by value (which would be wrong for a container type like
// mdarray).

}

template<class in_object_1_t,
         class in_object_2_t,
         class out_object_t>
void add(in_object_1_t x,
         in_object_2_t y,
         out_object_t z)
{
  if constexpr (z.rank() == 1) {
    add_rank_1 (x, y, z);
  }
  else if constexpr (z.rank() == 2) {
    add_rank_2 (x, y, z);
  }
  else {
    static_assert("Not implemented");
  }
}

template<class ExecutionPolicy,
         class in_object_1_t,
         class in_object_2_t,
         class out_object_t>
void add(ExecutionPolicy&& /* exec */,
         in_object_1_t x,
         in_object_2_t y,
         out_object_t z)
{
  add(x, y, z);
}

} // end namespace linalg
} // end inline namespace __p1673_version_0
} // end namespace experimental
} // end namespace std

#endif //LINALG_INCLUDE_EXPERIMENTAL___P1673_BITS_BLAS1_LINALG_ADD_HPP_
