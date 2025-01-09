//@HEADER
// ************************************************************************
//
//                        Kokkos v. 4.0
//       Copyright (2022) National Technology & Engineering
//               Solutions of Sandia, LLC (NTESS).
//
// Under the terms of Contract DE-NA0003525 with NTESS,
// the U.S. Government retains certain rights in this software.
//
// Part of Kokkos, under the Apache License v2.0 with LLVM Exceptions.
// See https://kokkos.org/LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// ************************************************************************
//@HEADER

#ifndef LINALG_INCLUDE_EXPERIMENTAL_BITS_LAYOUT_TRIANGLE_HPP_
#define LINALG_INCLUDE_EXPERIMENTAL_BITS_LAYOUT_TRIANGLE_HPP_

#include "layout_tags.hpp"

#include <type_traits>
#include <cstdint>

namespace MDSPAN_IMPL_STANDARD_NAMESPACE {
namespace MDSPAN_IMPL_PROPOSED_NAMESPACE {
inline namespace __p1673_version_0 {
namespace linalg {

namespace __triangular_layouts_impl {

template <class, class, class, class, class, class>
struct __lower_triangle_layout_impl;

// FIXME work-around for #4.
#if 0

// lower triangular offsets are triangular numbers (n*(n+1)/2)
template <
  ptrdiff_t ExtLast, ptrdiff_t... Exts, class BaseMap, class LastTwoMap,
  size_t... ExtIdxs, size_t... ExtMinus2Idxs
>
struct __lower_triangle_layout_impl<
  extents<Exts..., ExtLast, ExtLast>,
  BaseMap, LastTwoMap,
  std::integer_sequence<size_t, ExtIdxs...>,
  std::integer_sequence<size_t, ExtMinus2Idxs...>
> {

private:

  static constexpr auto __rank = sizeof...(Exts) + 2;

  _MDSPAN_NO_UNIQUE_ADDRESS LastTwoMap _trimap;
  _MDSPAN_NO_UNIQUE_ADDRESS BaseMap _base_map;

public:


  template <class... Integral>
  MDSPAN_FORCE_INLINE_FUNCTION
  constexpr ptrdiff_t operator()(Integral... idxs) const noexcept {
    auto base_val = _base_map(
      [&](size_t N) {
        _MDSPAN_FOLD_PLUS_RIGHT(((ExtIdxs == N) ? idx : 0), /* + ... + */ 0)
      }(ExtMinus2Idxs)...
    );
    auto triang_val = _trimap(
      _MDSPAN_FOLD_PLUS_RIGHT(((ExtIdxs == __rank - 2) ? idx : 0), /* + ... + */ 0),
      _MDSPAN_FOLD_PLUS_RIGHT(((ExtIdxs == __rank - 1) ? idx : 0), /* + ... + */ 0)
    );
    return base_val * triang_val;
  }

};

#endif // 0  

} // end namespace __triangular_layouts_impl

template <class Triangle, class StorageOrder>
class layout_blas_packed;

} // end namespace linalg
} // end inline namespace __p1673_version_0
} // end namespace MDSPAN_IMPL_PROPOSED_NAMESPACE
} // end namespace MDSPAN_IMPL_STANDARD_NAMESPACE

#endif //LINALG_INCLUDE_EXPERIMENTAL_BITS_LAYOUT_TRIANGLE_HPP_
