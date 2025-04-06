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

#ifndef LINALG_INCLUDE_EXPERIMENTAL___P1673_BITS_MAYBE_STATIC_SIZE_HPP_
#define LINALG_INCLUDE_EXPERIMENTAL___P1673_BITS_MAYBE_STATIC_SIZE_HPP_

#include <mdspan/mdspan.hpp>

namespace MDSPAN_IMPL_STANDARD_NAMESPACE {
namespace MDSPAN_IMPL_PROPOSED_NAMESPACE {
inline namespace __p1673_version_0 {
namespace linalg {
namespace impl {

template <class T, T Value, T DynSentinel>
struct __maybe_static_value {

  MDSPAN_INLINE_FUNCTION constexpr
  __maybe_static_value(T) noexcept { }
  MDSPAN_INLINE_FUNCTION MDSPAN_IMPL_CONSTEXPR_14
  __maybe_static_value& operator=(T) noexcept { }

  MDSPAN_INLINE_FUNCTION_DEFAULTED constexpr
  __maybe_static_value() noexcept = default;
  MDSPAN_INLINE_FUNCTION_DEFAULTED constexpr
  __maybe_static_value(__maybe_static_value const&) noexcept = default;
  MDSPAN_INLINE_FUNCTION_DEFAULTED constexpr
  __maybe_static_value(__maybe_static_value&&) noexcept = default;
  MDSPAN_INLINE_FUNCTION_DEFAULTED MDSPAN_IMPL_CONSTEXPR_14_DEFAULTED
  __maybe_static_value& operator=(__maybe_static_value const&) noexcept = default;
  MDSPAN_INLINE_FUNCTION_DEFAULTED MDSPAN_IMPL_CONSTEXPR_14_DEFAULTED
  __maybe_static_value& operator=(__maybe_static_value&&) noexcept = default;
  MDSPAN_INLINE_FUNCTION_DEFAULTED
  ~__maybe_static_value() = default;

  static constexpr auto value = Value;
  static constexpr auto is_static = true;
  static constexpr auto value_static = Value;
};

template <class T, T DynSentinel>
struct __maybe_static_value<T, DynSentinel, DynSentinel> {
  T value{};
  static constexpr auto is_static = false;
  static constexpr auto value_static = DynSentinel;
};

template <::std::size_t StaticSize, ::std::size_t Sentinel=dynamic_extent>
using __maybe_static_extent = __maybe_static_value<::std::size_t, StaticSize, Sentinel>;

} // end namespace impl
} // end namespace linalg
} // end inline namespace __p1673_version_0
} // end namespace MDSPAN_IMPL_PROPOSED_NAMESPACE
} // end namespace MDSPAN_IMPL_STANDARD_NAMESPACE

#endif //LINALG_INCLUDE_EXPERIMENTAL___P1673_BITS_MAYBE_STATIC_SIZE_HPP_
