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

#include "__p1673_bits/linalg_config.h"
#include <complex>
#include <mdspan/mdspan.hpp>
#include <type_traits>
#ifdef LINALG_HAS_CBLAS
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

// The padded layouts are class templates taking a size_t, so detecting them
// takes a partial specialization rather than is_same_v.
template<class Layout>
constexpr bool is_padded_layout_v = false;

template<std::size_t PaddingValue>
constexpr bool is_padded_layout_v<layout_left_padded<PaddingValue>> = true;

template<std::size_t PaddingValue>
constexpr bool is_padded_layout_v<layout_right_padded<PaddingValue>> = true;

template<class Layout>
constexpr bool is_blas_layout_type_v =
  // Assume that we have a C BLAS, which accepts
  // both row-major and column-major layouts.
  //
  // This just means that the layouts COULD be valid.
  // For layout_stride, we need to check the strides first.
  std::is_same_v<Layout, layout_left> ||
  std::is_same_v<Layout, layout_right> ||
  is_padded_layout_v<Layout> ||
  std::is_same_v<Layout, layout_stride>;

// The BLAS accepts accessors that deal with pointers to memory.
// default_accessor is a class template, so we can't just use is_same_v directly.
//
// scale doesn't accept conjugated_accessor or scaled_accessor
// because those are read-only accessors, and scale needs to
// write to the mdspan's elements.

template<class Accessor>
constexpr bool is_default_accessor_v = false;

template<class ElementType>
constexpr bool is_default_accessor_v<default_accessor<ElementType>> = true;

template<class Accessor>
constexpr bool is_blas_accessor_type_v =
  is_default_accessor_v<Accessor>;

#ifdef LINALG_HAS_CBLAS
// Deduce the BLAS integer index type from the first parameter
// (the length N) of cblas_dscal, as declared by the cblas.h in scope:
// int on an LP64 build, a 64-bit integer on an ILP64 build.
template<class R, class A, class... Rest> A first_param(R (*)(A, Rest...));
using cblas_index = decltype(first_param(cblas_dscal));
#endif

// We made the above queries traits, with their typical `_v` prefix.
// We make maybe_can_blas_scale() a function.
// That's a matter of taste; it could be a trait too.

} // end namespace impl
} // end namespace linalg
} // end inline namespace __p1673_version_0
} // end namespace MDSPAN_IMPL_PROPOSED_NAMESPACE
} // end namespace MDSPAN_IMPL_STANDARD_NAMESPACE

#endif //LINALG_INCLUDE_EXPERIMENTAL___P1673_BITS_BLAS_HELPERS_HPP_
