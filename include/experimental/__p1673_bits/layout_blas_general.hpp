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

#ifndef LINALG_INCLUDE_EXPERIMENTAL___P1673_BITS_LAYOUT_BLAS_GENERAL_HPP_
#define LINALG_INCLUDE_EXPERIMENTAL___P1673_BITS_LAYOUT_BLAS_GENERAL_HPP_

#include "maybe_static_size.hpp"
#include "layout_tags.hpp"

#include <experimental/__p0009_bits/macros.hpp>
#include <experimental/__p0009_bits/layout_left.hpp>
#include <experimental/__p0009_bits/layout_right.hpp>

namespace std {
namespace experimental {
inline namespace __p1673_version_0 {

namespace __layout_blas_general_impl {

// Strides is an extents object.
//
// Key is that the base mapping uses the strides, not the extents.
//
// For BaseLayout=LayoutLeft, the leftmost stride is 1; for
// BaseLayout=LayoutRight, the rightmost stride is 1.  Even though
// this is a compile-time constant, implementations might choose not
// to store it, by making Strides one shorter than Extents.  The
// implementation below would need to change if
template <class BaseLayout, class Extents, class Strides>
class __layout_blas_impl {
private:
  _MDSPAN_NO_UNIQUE_ADDRESS Extents _extents;
  using mapping_type = BaseLayout::template mapping<Strides>;
  _MDSPAN_NO_UNIQUE_ADDRESS mapping_type _mapping;

private:
  template <class, class, class>
  friend class __layout_blas_impl;

public:

  MDSPAN_INLINE_FUNCTION_DEFAULTED constexpr __layout_blas_impl() noexcept = default;
  MDSPAN_INLINE_FUNCTION_DEFAULTED constexpr __layout_blas_impl(__layout_blas_impl const&) noexcept = default;
  MDSPAN_INLINE_FUNCTION_DEFAULTED constexpr __layout_blas_impl(__layout_blas_impl&&) noexcept = default;
  MDSPAN_INLINE_FUNCTION_DEFAULTED _MDSPAN_CONSTEXPR_14_DEFAULTED __layout_blas_impl& operator=(__layout_blas_impl const&) noexcept = default;
  MDSPAN_INLINE_FUNCTION_DEFAULTED _MDSPAN_CONSTEXPR_14_DEFAULTED __layout_blas_impl& operator=(__layout_blas_impl&&) noexcept = default;
  MDSPAN_INLINE_FUNCTION_DEFAULTED ~__layout_blas_impl() = default;

  MDSPAN_INLINE_FUNCTION
  constexpr explicit
  __layout_blas_impl(Extents const& base_extents) noexcept
    : _extents(base_extents),
      _mapping(base_extents)
  { }

  MDSPAN_INLINE_FUNCTION
  constexpr explicit
  __layout_blas_impl(Extents const& base_extents, Strides const& strides) noexcept
    : _extents(base_extents),
      _mapping(strides) // key is that the base mapping uses the strides, not the extents.
  { }

  template<class OtherExtents, class OtherStrides>
  MDSPAN_INLINE_FUNCTION _MDSPAN_CONSTEXPR_14
  __layout_blas_impl& operator=(
    __layout_blas_impl<BaseLayout, OtherExtents, OtherStrides> const& other)
  {
    this->_extents = other.extents();
    this->_mapping = other._mapping;
    return *this;
  }

  template <class... Integral>
  MDSPAN_FORCE_INLINE_FUNCTION
  constexpr ptrdiff_t operator()(Integral... idxs) const noexcept {
    return _mapping(idxs...);
  }

  MDSPAN_INLINE_FUNCTION constexpr bool is_unique() const noexcept { return true; }
  MDSPAN_INLINE_FUNCTION constexpr bool is_contiguous() const noexcept { return _mapping.extents () == _extents; }
  MDSPAN_INLINE_FUNCTION constexpr bool is_strided() const noexcept { return true; }

  MDSPAN_INLINE_FUNCTION static constexpr bool is_always_unique() noexcept { return true; }
  MDSPAN_INLINE_FUNCTION static constexpr bool is_always_contiguous() noexcept { return false; /* FIXME */ }
  MDSPAN_INLINE_FUNCTION static constexpr bool is_always_strided() noexcept { return true; }

  MDSPAN_INLINE_FUNCTION constexpr Extents extents() const noexcept { return _extents; }

  MDSPAN_INLINE_FUNCTION
  constexpr ptrdiff_t required_span_size() const noexcept {
    return _mapping.required_span_size();
  }

  MDSPAN_INLINE_FUNCTION
  constexpr ptrdiff_t stride(size_t r) const noexcept {
    return _mapping.stride(r);
  }

  template<class OtherExtents, class OtherStrides>
  MDSPAN_INLINE_FUNCTION
  friend constexpr bool operator==(
    __layout_blas_impl const& a,
    __layout_blas_impl<BaseLayout, OtherExtents, OtherStrides> const& b) noexcept {
    return a.extents() == b.extents() &&
      a._mapping.extents() == b._mapping.extents();
  }

  template<class OtherExtents, class OtherStrides>
  MDSPAN_INLINE_FUNCTION
  friend constexpr bool operator!=(
    __layout_blas_impl const& a,
    __layout_blas_impl<BaseLayout, OtherExtents, OtherStrides> const& b) noexcept {
    return ! (a == b);
  }

  // // Needed to work with subspan()
  // template <size_t N>
  // struct __static_stride_workaround {
  //   static constexpr ptrdiff_t value = __lda_t::is_static ?
  //     (BaseLayout::template __static_stride_workaround<N>::value == dynamic_extent ? dynamic_extent :
  //       (__lda_t::value_static * BaseLayout::template __static_stride_workaround<N>::value)
  //     ) : dynamic_extent;
  };
};


template <ptrdiff... OriginalExtents>
constexpr auto compute_column_major_strides_from_extents (const extents<OriginalExtents...> in);

         
template <ptrdiff... Extents>
constexpr extents<ptrdiff_t(1), Extents...>
prepend_one_to_extents (const extents<Extents...> in) {
  return extents<ptrdiff_t(1), Extents...>;
}

template <ptrdiff... Extents>
constexpr extents<Extents..., ptrdiff_t(1)>
append_one_to_extents (const extents<Extents...> in) {
  return extents<ptrdiff_t(1), Extents...>;
}

} // end namespace __layout_blas_general_impl

template <class StorageOrder>
class layout_blas_general;

template <>
class layout_blas_general<column_major_t> {
public:
  template <class Extents>
  using mapping = __layout_blas_general_impl::__layout_blas_impl<layout_left, Extents, Strides>;
};

template <>
class layout_blas_general<row_major_t> {
public:
  template <class Extents>
  using mapping = __layout_blas_general_impl::__layout_blas_impl<layout_right, Extents, Strides>;
};

} // end inline namespace __p1673_version_0
} // end namespace experimental
} // end namespace std


#endif //LINALG_INCLUDE_EXPERIMENTAL___P1673_BITS_LAYOUT_BLAS_GENERAL_HPP_
