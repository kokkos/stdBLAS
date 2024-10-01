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

#include <mdspan/mdspan.hpp>

namespace MDSPAN_IMPL_STANDARD_NAMESPACE {
namespace MDSPAN_IMPL_PROPOSED_NAMESPACE {
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
    using type = extents<typename Extents::index_type, Extents::static_extent(1), Extents::static_extent(0)>;
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
      typename transpose_extents_t<Extents>::index_type,
      typename Extents::index_type>, "Please fix transpose_extents_t to account "
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
  using nested_layout_type = Layout;

  template<class Extents>
  struct mapping {
  private:
    using nested_mapping_type =
      typename Layout::template mapping<impl::transpose_extents_t<Extents>>;

  public:
    using extents_type = Extents;
    using index_type = typename extents_type::index_type;
    using size_type = typename extents_type::size_type;
    using rank_type = typename extents_type::rank_type;
    using layout_type = layout_transpose;

    constexpr explicit mapping(const nested_mapping_type& map)
      : nested_mapping_(map),
        extents_(impl::transpose_extents(map.extents()))
    {}

    constexpr const extents_type& extents() const noexcept
    {
      return extents_;
    }

    constexpr index_type required_span_size() const
      noexcept(noexcept(nested_mapping_.required_span_size()))
    {
      return nested_mapping_.required_span_size();
    }

    template<class IndexType0, class IndexType1>
      requires(std::is_convertible_v<IndexType0, index_type> &&
               std::is_convertible_v<IndexType1, index_type>)
    index_type operator() (IndexType0 i, IndexType1 j) const
    {
      return nested_mapping_(j, i);
    }

    const nested_mapping_type& nested_mapping() const
    {
      return nested_mapping_;
    }

    static constexpr bool is_always_unique() noexcept {
      return nested_mapping_type::is_always_unique();
    }
    static constexpr bool is_always_exhaustive() noexcept {
      return nested_mapping_type::is_always_contiguous();
    }
    static constexpr bool is_always_strided() noexcept {
      return nested_mapping_type::is_always_strided();
    }

    constexpr bool is_unique() const
    {
      return nested_mapping_.is_unique();
    }
    constexpr bool is_exhaustive() const
    {
      return nested_mapping_.is_exhaustive();
    }
    constexpr bool is_strided() const
    {
      return nested_mapping_.is_strided();
    }

    constexpr index_type stride(size_t r) const
    {
      assert(this->is_strided());
      assert(r < extents_type::rank());
      return nested_mapping_.stride(r == 0 ? 1 : 0);
    }

    template<class OtherExtents>
    friend constexpr bool
    operator==(const mapping& lhs, const mapping<OtherExtents>& rhs) noexcept
    {
      return lhs.nested_mapping_ == rhs.nested_mapping_;
    }

  private:
    nested_mapping_type nested_mapping_;
    extents_type extents_;
  };
};

namespace impl {

  template<class ElementType, class Accessor>
  struct transposed_element_accessor
  {
    using element_type = ElementType;
    using accessor_type = Accessor;

    static accessor_type accessor(const Accessor& a) { return accessor_type(a); }
  };

  template<class ElementType>
  struct transposed_element_accessor<
    ElementType, default_accessor<ElementType>>
  {
    using element_type = ElementType;
    using accessor_type = default_accessor<element_type>;

    static accessor_type accessor(const default_accessor<ElementType>& a) { return accessor_type(a); }
  };

  template<class Layout>
  struct transposed_layout {
    using layout_type = layout_transpose<Layout>;

    template<class OriginalMapping>
    static auto mapping(const OriginalMapping& orig_map) {
      using extents_type = transpose_extents_t<typename OriginalMapping::extents_type>;
      using return_mapping_type = typename layout_type::template mapping<extents_type>;
      return return_mapping_type{orig_map};
    }
  };

  template<>
  struct transposed_layout<layout_left> {
    using layout_type = layout_right;

    template<class OriginalExtents>
    static auto mapping(const typename layout_left::template mapping<OriginalExtents>& orig_map) {
      using original_mapping_type = typename layout_left::template mapping<OriginalExtents>;
      using extents_type = transpose_extents_t<typename original_mapping_type::extents_type>;
      using return_mapping_type = typename layout_type::template mapping<extents_type>;
      return return_mapping_type{transpose_extents(orig_map.extents())};
    }
  };

  template<>
  struct transposed_layout<layout_right> {
    using layout_type = layout_left;

    template<class OriginalExtents>
    static auto mapping(const typename layout_right::template mapping<OriginalExtents>& orig_map) {
      using original_mapping_type = typename layout_right::template mapping<OriginalExtents>;
      using extents_type = transpose_extents_t<typename original_mapping_type::extents_type>;
      using return_mapping_type = typename layout_type::template mapping<extents_type>;
      return return_mapping_type{transpose_extents(orig_map.extents())};
    }
  };

  template<>
  struct transposed_layout<layout_stride> {
    using layout_type = layout_stride;

    template<class OriginalExtents>
    static auto mapping(const typename layout_stride::template mapping<OriginalExtents>& orig_map) {
      using original_mapping_type = typename layout_stride::template mapping<OriginalExtents>;
      // MSVC 2022 doesn't like the following commented-out line of code.
      // See https://github.com/kokkos/stdBLAS/issues/242#issuecomment-1174738571
      //
      //using extents_type = transpose_extents_t<typename original_mapping_type::extents_type>;
      using original_extents_type = typename original_mapping_type::extents_type;
      using extents_type = transpose_extents_t<original_extents_type>;
      using return_mapping_type = typename layout_type::template mapping<extents_type>;
      // NOTE (mfh 2022/07/04) Commented-out code relates
      // to the build error reported in my comment here:
      //
      // https://github.com/kokkos/stdBLAS/issues/242
      return return_mapping_type{
	transpose_extents(orig_map.extents()),
	std::array<typename extents_type::index_type, OriginalExtents::rank() /* orig_map.rank() */ >{
	  orig_map.stride(1),
	  orig_map.stride(0)}};
    }
  };

#if defined(LINALG_FIX_TRANSPOSED_FOR_PADDED_LAYOUTS)
  template<size_t PaddingValue>
  struct transposed_layout<layout_left_padded<PaddingValue>> {
    using layout_type = layout_right_padded<PaddingValue>;

    template<class OriginalExtents>
    static auto mapping(const typename layout_left_padded<PaddingValue>::template mapping<OriginalExtents>& orig_map) {
      using input_mapping_type =
        typename layout_left_padded<PaddingValue>::template mapping<OriginalExtents>;
      using output_extents_type =
        transpose_extents_t<typename input_mapping_type::extents_type>;
      using output_mapping_type =
        typename layout_type::template mapping<output_extents_type>;

      const auto padding_value = orig_map.stride(1);
      return output_mapping_type{
	transpose_extents(orig_map.extents()),
        padding_value
      };
    }
  };

  template<size_t PaddingValue>
  struct transposed_layout<layout_right_padded<PaddingValue>> {
    using layout_type = layout_left_padded<PaddingValue>;

    template<class OriginalExtents>
    static auto mapping(const typename layout_right_padded<PaddingValue>::template mapping<OriginalExtents>& orig_map) {
      using input_mapping_type =
        typename layout_right_padded<PaddingValue>::template mapping<OriginalExtents>;
      using output_extents_type =
        transpose_extents_t<typename input_mapping_type::extents_type>;
      using output_mapping_type =
        typename layout_type::template mapping<output_extents_type>;

      const auto padding_value = orig_map.stride(0);
      return output_mapping_type{
	transpose_extents(orig_map.extents()),
        padding_value
      };
    }
  };
#endif // LINALG_FIX_TRANSPOSED_FOR_PADDED_LAYOUTS
  
  template<class StorageOrder>
  using opposite_storage_t = std::conditional_t<
    std::is_same_v<StorageOrder, column_major_t>,
    row_major_t,
    column_major_t>;

  template<class StorageOrder>
  struct transposed_layout<layout_blas_general<StorageOrder>> {
    using layout_type = layout_blas_general<
      opposite_storage_t<StorageOrder>>;

    template<class OriginalExtents>
    static auto mapping(const typename layout_blas_general<StorageOrder>::template mapping<OriginalExtents>& orig_map) {
      using original_mapping_type = typename layout_blas_general<StorageOrder>::template mapping<OriginalExtents>;
      using extents_type = transpose_extents_t<typename original_mapping_type::extents_type>;
      using return_mapping_type = typename layout_type::template mapping<extents_type>;
      const auto whichStride = std::is_same_v<StorageOrder, column_major_t> ? orig_map.stride(1) : orig_map.stride(0);
      return return_mapping_type{transpose_extents(orig_map.extents()), whichStride};
    }
  };

  template<class Triangle>
  using opposite_triangle_t = std::conditional_t<
    std::is_same_v<Triangle, upper_triangle_t>,
    lower_triangle_t,
    upper_triangle_t>;

  template<class Triangle, class StorageOrder>
  struct transposed_layout<layout_blas_packed<Triangle, StorageOrder>> {
    using layout_type = layout_blas_packed<
      opposite_triangle_t<Triangle>,
      opposite_storage_t<StorageOrder>>;

    template<class OriginalExtents>
    static auto mapping(const typename layout_blas_packed<Triangle, StorageOrder>::template mapping<OriginalExtents>& orig_map) {
      using original_mapping_type = typename layout_blas_packed<Triangle, StorageOrder>::template mapping<OriginalExtents>;
      using extents_type = transpose_extents_t<typename original_mapping_type::extents_type>;
      using return_mapping_type = typename layout_type::template mapping<extents_type>;
      return return_mapping_type{transpose_extents(orig_map.extents())};
    }
  };

  template<class NestedLayout>
  struct transposed_layout<layout_transpose<NestedLayout>> {
    using layout_type = NestedLayout;
  };
} // namespace impl

template<class ElementType, class Extents, class Layout, class Accessor>
auto transposed(mdspan<ElementType, Extents, Layout, Accessor> a)
{
  using element_type = typename impl::transposed_element_accessor<ElementType, Accessor>::element_type;
  using layout_type = typename impl::transposed_layout<Layout>::layout_type;
  using accessor_type = typename impl::transposed_element_accessor<ElementType, Accessor>::accessor_type;

  auto mapping = impl::transposed_layout<Layout>::mapping(a.mapping());
  auto accessor = impl::transposed_element_accessor<ElementType, Accessor>::accessor(a.accessor());
  return mdspan<element_type, typename decltype(mapping)::extents_type, layout_type, accessor_type>{a.data_handle(), mapping, accessor};
}

} // end namespace linalg
} // end inline namespace __p1673_version_0
} // end namespace MDSPAN_IMPL_PROPOSED_NAMESPACE
} // end namespace MDSPAN_IMPL_STANDARD_NAMESPACE

#endif //LINALG_INCLUDE_EXPERIMENTAL___P1673_BITS_TRANSPOSED_HPP_
