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
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA,
OR
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

namespace impl {
  // This struct helps us impose the rank constraint
  // on the type alias itself.
  MDSPAN_TEMPLATE_REQUIRES(
    class Extents,
    /* requires */ (Extents::rank() == 2)
  )
  struct transpose_extents_t_impl
  {
    using type = extents<Extents::static_extent(1), Extents::static_extent(0)>;
  };

  template<class Extents>
  using transpose_extents_t = typename transpose_extents_t_impl<Extents>::type;

  MDSPAN_TEMPLATE_REQUIRES(
    class Extents,
    /* requires */ (Extents::rank() == 2)
  )
  transpose_extents_t<Extents> transpose_extents(const Extents& e)
  {
    static_assert(std::is_same_v<
      typename transpose_extents_t<Extents>::size_type,
      typename Extents::size_type>, "Please fix transpose_extents_t to account "
      "for P2553, which adds a template parameter SizeType to extents.");

    constexpr size_t ext0 = Extents::static_extent(0);
    constexpr size_t ext1 = Extents::static_extent(1);

    if constexpr (ext0 == dynamic_extent) {
      if constexpr (ext1 == dynamic_extent) {
	return transpose_extents_t<Extents>{e.extent(1), e.extent(0)};
      } else {
	return transpose_extents_t<Extents>{/* e.extent(1), */ e.extent(0)};
      }
    } else {
      if constexpr (ext1 == dynamic_extent) {
	return transpose_extents_t<Extents>{e.extent(1) /* , e.extent(0) */ };
      } else {
	return transpose_extents_t<Extents>{}; // all extents are static
      }
    }
  }
}

template<class Layout>
class layout_transpose {
public:
  template<class Extents>
  struct mapping {
  private:
    using nested_mapping_type =
      typename Layout::template mapping<impl::transpose_extents_t<Extents>>;
    nested_mapping_type nested_mapping_;

  public:
    using extents_type = Extents;
    using size_type = typename extents_type::size_type;
    using layout_type = layout_transpose;

    constexpr explicit mapping(const nested_mapping_type& map)
      : nested_mapping_(map) {}

    constexpr extents_type extents() const
      noexcept(noexcept(nested_mapping_.extents()))
    {
      return impl::transpose_extents(nested_mapping_.extents());
    }

    constexpr size_type required_span_size() const
      noexcept(noexcept(nested_mapping_.required_span_size()))
    {
      return nested_mapping_.required_span_size();
    }

    template<class IndexType, class... Indices>
    typename Extents::size_type operator() (Indices... rest, IndexType i, IndexType j) const
      noexcept(noexcept(nested_mapping_(rest..., j, i)))
    {
      return nested_mapping_(rest..., j, i);
    }

    nested_mapping_type nested_mapping() const
    {
      return nested_mapping_;
    }

    static constexpr bool is_always_unique() {
      return nested_mapping_type::is_always_unique();
    }
    static constexpr bool is_always_contiguous() {
      return nested_mapping_type::is_always_contiguous();
    }
    static constexpr bool is_always_strided() {
      return nested_mapping_type::is_always_strided();
    }

    constexpr bool is_unique() const
      noexcept(noexcept(nested_mapping_.is_unique()))
    {
      return nested_mapping_.is_unique();
    }
    constexpr bool is_contiguous() const
      noexcept(noexcept(nested_mapping_.is_contiguous()))
    {
      return nested_mapping_.is_contiguous();
    }
    constexpr bool is_strided() const
      noexcept(noexcept(nested_mapping_.is_strided()))
    {
      return nested_mapping_.is_strided();
    }

    constexpr size_type stride(size_t r) const
      noexcept(noexcept(nested_mapping_.stride(r)))
    {
      if (r == extents_type::rank() - 1) {
	return nested_mapping_.stride(extents_type::rank() - 2);
      }
      else if (r == extents_type::rank() - 2) {
	return nested_mapping_.stride(extents_type::rank() - 1);
      }
      else {
	return nested_mapping_.stride(r);
      }
    }

    template<class OtherExtents>
    friend constexpr bool
    operator==(const mapping& lhs, const mapping<OtherExtents>& rhs) noexcept
    {
      return lhs.nested_mapping_ == rhs.nested_mapping_;
    }
  };
};

template<class EltType, class Extents, class Layout, class Accessor>
mdspan<EltType, Extents, layout_transpose<Layout>, Accessor>
transposed(mdspan<EltType, Extents, Layout, Accessor> a)
{
  using layout_type = layout_transpose<Layout>;
  using mapping_type = typename layout_type::template mapping<Extents>;
  return mdspan<EltType, Extents, layout_type, Accessor> (
    a.data (), mapping_type (a.mapping ()), a.accessor ());
}

} // end namespace linalg
} // end inline namespace __p1673_version_0
} // end namespace experimental
} // end namespace std

#endif //LINALG_INCLUDE_EXPERIMENTAL___P1673_BITS_TRANSPOSED_HPP_
