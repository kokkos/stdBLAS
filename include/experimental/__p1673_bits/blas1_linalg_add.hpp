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

// FIXME (Hoemmen 2021/05/28) Latest version of P0009 (mdspan) uses size_t
// instead of ptrdiff_t, but the implementation hasn't changed yet.
template<class ElementType_x,
         ptrdiff_t ext_x,
         class Layout_x,
         class Accessor_x,
         class ElementType_y,
         ptrdiff_t ext_y,
         class Layout_y,
         class Accessor_y,
         class ElementType_z,
         ptrdiff_t ext_z,
         class Layout_z,
         class Accessor_z>
void add_rank_1(
  std::experimental::basic_mdspan<ElementType_x, std::experimental::extents<ext_x>, Layout_x, Accessor_x> x,
  std::experimental::basic_mdspan<ElementType_y, std::experimental::extents<ext_y>, Layout_y, Accessor_y> y,
  std::experimental::basic_mdspan<ElementType_z, std::experimental::extents<ext_z>, Layout_z, Accessor_z> z)
{
  for (ptrdiff_t i = 0; i < z.extent(0); ++i) {
    z(i) = x(i) + y(i);
  }
}

template<class ElementType_x,
         ptrdiff_t numRows_x, ptrdiff_t numCols_x,
         class Layout_x,
         class Accessor_x,
         class ElementType_y,
         ptrdiff_t numRows_y, ptrdiff_t numCols_y,
         class Layout_y,
         class Accessor_y,
         class ElementType_z,
         ptrdiff_t numRows_z, ptrdiff_t numCols_z,
         class Layout_z,
         class Accessor_z>
void add_rank_2(
  std::experimental::basic_mdspan<ElementType_x, std::experimental::extents<numRows_x, numCols_x>, Layout_x, Accessor_x> x,
  std::experimental::basic_mdspan<ElementType_y, std::experimental::extents<numRows_y, numCols_y>, Layout_y, Accessor_y> y,
  std::experimental::basic_mdspan<ElementType_z, std::experimental::extents<numRows_z, numCols_z>, Layout_z, Accessor_z> z)
{
  for (ptrdiff_t j = 0; j < x.extent(1); ++j) {
    for (ptrdiff_t i = 0; i < x.extent(0); ++i) {
      z(i,j) = x(i,j) + y(i,j);
    }
  }
}

} // end anonymous namespace

template<class ElementType_x,
         ptrdiff_t ... ext_x,
         class Layout_x,
         class Accessor_x,
         class ElementType_y,
         ptrdiff_t ... ext_y,
         class Layout_y,
         class Accessor_y,
         class ElementType_z,
         ptrdiff_t ... ext_z,
         class Layout_z,
         class Accessor_z>
  requires (sizeof...(ext_x) == sizeof...(ext_y) && sizeof...(ext_x) == sizeof...(ext_z))
void add(  
  std::experimental::basic_mdspan<ElementType_x, std::experimental::extents<ext_x ...>, Layout_x, Accessor_x> x,
  std::experimental::basic_mdspan<ElementType_y, std::experimental::extents<ext_y ...>, Layout_y, Accessor_y> y,
  std::experimental::basic_mdspan<ElementType_z, std::experimental::extents<ext_z ...>, Layout_z, Accessor_z> z)
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
         class ElementType_x,
         ptrdiff_t ... ext_x,
         class Layout_x,
         class Accessor_x,
         class ElementType_y,
         ptrdiff_t ... ext_y,
         class Layout_y,
         class Accessor_y,
         class ElementType_z,
         ptrdiff_t ... ext_z,
         class Layout_z,
         class Accessor_z>
  requires (sizeof...(ext_x) == sizeof...(ext_y) && sizeof...(ext_x) == sizeof...(ext_z))
void add(
  ExecutionPolicy&& /* exec */,
  std::experimental::basic_mdspan<ElementType_x, std::experimental::extents<ext_x ...>, Layout_x, Accessor_x> x,
  std::experimental::basic_mdspan<ElementType_y, std::experimental::extents<ext_y ...>, Layout_y, Accessor_y> y,
  std::experimental::basic_mdspan<ElementType_z, std::experimental::extents<ext_z ...>, Layout_z, Accessor_z> z)
{
  add(x, y, z);
}

} // end namespace linalg
} // end inline namespace __p1673_version_0
} // end namespace experimental
} // end namespace std

#endif //LINALG_INCLUDE_EXPERIMENTAL___P1673_BITS_BLAS1_LINALG_ADD_HPP_
