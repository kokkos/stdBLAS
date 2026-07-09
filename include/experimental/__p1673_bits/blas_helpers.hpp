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

#ifndef LINALG_INCLUDE_EXPERIMENTAL___P1673_BITS_BLAS_HELPERS_HPP_
#define LINALG_INCLUDE_EXPERIMENTAL___P1673_BITS_BLAS_HELPERS_HPP_

#include <complex>
#include <mdspan/mdspan.hpp>
#include <type_traits>
#ifdef KOKKOS_ENABLE_CBLAS
#include "cblas.h"
#endif

namespace MDSPAN_IMPL_STANDARD_NAMESPACE {
namespace MDSPAN_IMPL_PROPOSED_NAMESPACE {
inline namespace __p1673_version_0 {
namespace linalg {
namespace impl {

template<class ValueType>
constexpr bool is_blas_value_type_v =
  std::is_same_v<ValueType, float> ||
  std::is_same_v<ValueType, double> ||
  std::is_same_v<ValueType, std::complex<float>> ||
  std::is_same_v<ValueType, std::complex<double>>;

template<class Layout>
constexpr bool is_blas_layout_type_v =
  // Assume that we have a C BLAS, which accepts
  // both row-major and column-major layouts.
  //
  // This just means that the layouts COULD be valid.
  // For layout_stride, we need to check the strides first.
  std::is_same_v<Layout, layout_left> ||
  std::is_same_v<Layout, layout_right> ||
  // AMK 5/20/26 - layout_left_padded and layout_right_padded were added in C++26.
  // According to cppreference, padded layouts are covered by the same feature test as submdspan.
  #ifdef __cpp_lib_submdspan
  std::is_same_v<Layout, layout_left_padded> ||
  std::is_same_v<Layout, layout_right_padded> ||
  #endif
  std::is_same_v<Layout, layout_stride>;

// The BLAS accepts accessors that deal with pointers to memory:
// default_accessor and aligned_accessor.
// Those are both templated classes, so we can't just use is_same_v directly.
//
// scale doesn't accept conjugated_accessor or scaled_accessor
// because those are read-only accessors, and scale needs to
// write to the mdspan's elements.

template<class Accessor>
constexpr bool is_default_accessor_v = false;

template<class ElementType>
constexpr bool is_default_accessor_v<default_accessor<ElementType>> = true;

template<class Accessor>
constexpr bool is_aligned_accessor_v = false;

#ifdef __cpp_lib_aligned_accessor
template<class ElementType, std::size_t ByteAlignment>
constexpr bool is_aligned_accessor_v<aligned_accessor<ElementType, ByteAlignment>> = true;
#endif

template<class Accessor>
constexpr bool is_blas_accessor_type_v =
  is_default_accessor_v<Accessor> ||
  is_aligned_accessor_v<Accessor>;

// We made the above queries traits, with their typical `_v` prefix.
// We make maybe_can_blas_scale() a function.
// That's a matter of taste; it could be a trait too.

} // end namespace impl
} // end namespace linalg
} // end inline namespace __p1673_version_0
} // end namespace MDSPAN_IMPL_PROPOSED_NAMESPACE
} // end namespace MDSPAN_IMPL_STANDARD_NAMESPACE

#endif //LINALG_INCLUDE_EXPERIMENTAL___P1673_BITS_BLAS_HELPERS_HPP_
