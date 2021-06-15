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

#ifndef LINALG_INCLUDE_EXPERIMENTAL___P1673_BITS_TRANSPOSED_HPP_
#define LINALG_INCLUDE_EXPERIMENTAL___P1673_BITS_TRANSPOSED_HPP_

#include <experimental/mdspan>

namespace std {
namespace experimental {
inline namespace __p1673_version_0 {
namespace linalg {

namespace {
  template<extents<>::size_type ext0, extents<>::size_type ext1>
  extents<ext1, ext0> transpose_extents (extents<ext0, ext1> e)
  {
    return extents<ext1, ext0> (e.extent(1), e.extent(0));
  }
}

template<class Layout>
class layout_transpose {
public:
  template<class Extents>
  struct mapping {
  private:
    using nested_mapping_type = typename Layout::template mapping<Extents>;
  public:
    nested_mapping_type nested_mapping;

    mapping() = default;

    mapping(nested_mapping_type map) : nested_mapping(map) {}

    // TODO insert other standard mapping things

    // for non-batched layouts
    Extents::size_type operator() (Extents::size_type i, Extents::size_type j) const {
      return nested_mapping(j, i);
    }

    constexpr auto extents() const noexcept {
      return transpose_extents(nested_mapping.extents());
    }

    constexpr bool is_unique() const noexcept {
      return nested_mapping.is_unique();
    }
    constexpr bool is_contiguous() const noexcept {
      return nested_mapping.is_contiguous();
    }
    constexpr bool is_strided() const noexcept {
      return nested_mapping.is_strided();
    }

    constexpr Extents::size_type stride(size_t r) const noexcept {
      // FIXME this only works for rank 2
      return nested_mapping.stride(size_t(1) - r);
    }

    // FIXME The overload below doesn't compile

    // // for batched layouts
    // ptrdiff_t operator() (ptrdiff_t... rest, ptrdiff_t i, ptrdiff_t j) const {
    //   return nested_mapping(rest..., j, i);
    // }
  };
};

template<class EltType, class Extents, class Layout, class Accessor>
basic_mdspan<EltType, Extents, layout_transpose<Layout>, Accessor>
transposed(basic_mdspan<EltType, Extents, Layout, Accessor> a)
{
  using layout_type = layout_transpose<Layout>;
  using mapping_type = typename layout_type::template mapping<Extents>;
  return basic_mdspan<EltType, Extents, layout_type, Accessor> (
    a.data (), mapping_type (a.mapping ()), a.accessor ());
}

} // end namespace linalg
} // end inline namespace __p1673_version_0
} // end namespace experimental
} // end namespace std

#endif //LINALG_INCLUDE_EXPERIMENTAL___P1673_BITS_TRANSPOSED_HPP_
