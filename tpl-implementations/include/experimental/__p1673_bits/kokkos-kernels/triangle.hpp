 /*
//@HEADER
// ************************************************************************
//
//                        Kokkos v. 2.0
//              Copyright (2019) Sandia Corporation
//
// Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
// the U.S. Government retains certain rights in this software.
//
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

#ifndef LINALG_TPLIMPLEMENTATIONS_INCLUDE_EXPERIMENTAL___P1673_BITS_KOKKOSKERNELS_TRIANGLE_UTILS_HPP_
#define LINALG_TPLIMPLEMENTATIONS_INCLUDE_EXPERIMENTAL___P1673_BITS_KOKKOSKERNELS_TRIANGLE_UTILS_HPP_

namespace KokkosKernelsSTD {
namespace Impl {

// Note: phrase it simply and the same as in specification ("has unique layout")
template <typename Layout,
          std::experimental::extents<>::size_type numRows,
          std::experimental::extents<>::size_type numCols>
constexpr bool is_unique_layout_v = Layout::template mapping<
    std::experimental::extents<numRows, numCols> >::is_always_unique();

template <typename Layout>
struct is_layout_blas_packed: public std::false_type {};

template <typename Triangle, typename StorageOrder>
struct is_layout_blas_packed<
  std::experimental::linalg::layout_blas_packed<Triangle, StorageOrder>>:
    public std::true_type {};

template <typename Layout>
constexpr bool is_layout_blas_packed_v = is_layout_blas_packed<Layout>::value;

// Note: will only signal failure for layout_blas_packed with different triangle
template <typename Layout, typename Triangle>
struct triangle_layout_match: public std::true_type {};

template <typename StorageOrder, typename Triangle1, typename Triangle2>
struct triangle_layout_match<
  std::experimental::linalg::layout_blas_packed<Triangle1, StorageOrder>,
  Triangle2>
{
  static constexpr bool value = std::is_same_v<Triangle1, Triangle2>;
};

template <typename Layout, typename Triangle>
constexpr bool triangle_layout_match_v = triangle_layout_match<Layout, Triangle>::value;

} // namespace Impl
} // namespace KokkosKernelsSTD
#endif
