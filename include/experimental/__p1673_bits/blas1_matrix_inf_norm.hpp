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

#ifndef LINALG_INCLUDE_EXPERIMENTAL___P1673_BITS_BLAS1_MATRIX_INF_NORM_HPP_
#define LINALG_INCLUDE_EXPERIMENTAL___P1673_BITS_BLAS1_MATRIX_INF_NORM_HPP_

#include <cmath>
#include <cstdlib>

namespace std {
namespace experimental {
inline namespace __p1673_version_0 {
namespace linalg {

  template<
    class ElementType,
    class Extents,
    class Layout,
    class Accessor,
    class Scalar>
Scalar matrix_inf_norm(
  std::experimental::basic_mdspan<ElementType, Extents, Layout, Accessor> A,
  Scalar init)
{
  using std::abs;
  using std::max;

  // Handle special cases.
  auto result = init;
  if (A.extent(0) == 0 || A.extent(1) == 0) {
    return result;
  }
  else if(A.extent(0) == ptrdiff_t(1) && A.extent(1) == ptrdiff_t(1)) {
    result += abs(A(0, 0));
    return result;
  }

  for (ptrdiff_t i = 0; i < A.extent(0); ++i) {
    auto row_sum = init;
    for (ptrdiff_t j = 0; j < A.extent(1); ++j) {
      row_sum += abs(A(i,j));
    }
    result = max(row_sum, result);
  }
  return result;
}

template<class ExecutionPolicy,
  class ElementType,
  class Extents,
  class Layout,
  class Accessor,
  class Scalar>
Scalar matrix_inf_norm(ExecutionPolicy&& /* exec */,
                       std::experimental::basic_mdspan<ElementType, Extents, Layout, Accessor> A,
                       Scalar init)
{
  return matrix_inf_norm(A, init);
}

namespace matrix_inf_norm_detail {

  using std::abs;
  
  // The point of this is to do correct ADL for abs,
  // without exposing "using std::abs" in the outer namespace.
  template<
    class ElementType,
    class Extents,
    class Layout,
    class Accessor>
  auto matrix_inf_norm_return_type_deducer(
    std::experimental::basic_mdspan<ElementType, Extents, Layout, Accessor> A) -> decltype(abs(A(0,0)));

} // namespace matrix_inf_norm_detail

template<
  class ElementType,
  class Extents,
  class Layout,
  class Accessor>
auto matrix_inf_norm(
  std::experimental::basic_mdspan<ElementType, Extents, Layout, Accessor> A)
-> decltype(matrix_inf_norm_detail::matrix_inf_norm_return_type_deducer(A))
{ 
  using return_t = decltype(matrix_inf_norm_detail::matrix_inf_norm_return_type_deducer(A));
  return matrix_inf_norm(A, return_t{});
}

template<class ExecutionPolicy,
         class ElementType,
         class Extents,
         class Layout,
         class Accessor>
auto matrix_inf_norm(
  ExecutionPolicy&& exec,
  std::experimental::basic_mdspan<ElementType, Extents, Layout, Accessor> A)
-> decltype(matrix_inf_norm_detail::matrix_inf_norm_return_type_deducer(A))
{
  using return_t = decltype(matrix_inf_norm_detail::matrix_inf_norm_return_type_deducer(A));
  return matrix_inf_norm(exec, A, return_t{});
}

} // end namespace linalg
} // end inline namespace __p1673_version_0
} // end namespace experimental
} // end namespace std

#endif //LINALG_INCLUDE_EXPERIMENTAL___P1673_BITS_BLAS1_MATRIX_INF_NORM_HPP_
