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

#ifndef LINALG_INCLUDE_EXPERIMENTAL___P1673_BITS_REAL_IF_NEEDED_HPP_
#define LINALG_INCLUDE_EXPERIMENTAL___P1673_BITS_REAL_IF_NEEDED_HPP_

#include <complex>
#include <type_traits>

namespace MDSPAN_IMPL_STANDARD_NAMESPACE {
namespace MDSPAN_IMPL_PROPOSED_NAMESPACE {
inline namespace __p1673_version_0 {
namespace linalg {
namespace impl{

template<class T, class = void>
struct has_real : std::false_type {};

// If I can find unqualified real via overload resolution,
// then assume that real(t) returns the real part of t.
template<class T>
struct has_real<T, decltype(real(std::declval<T>()), void())> : std::true_type {};

template<class T>
T real_if_needed_impl(const T& t, std::false_type)
{
  // If real(t) can't be ADL-found, then assume
  // that T represents a noncomplex number type.
  return t;
}

template<class T>
auto real_if_needed_impl(const T& t, std::true_type)
{
  if constexpr (std::is_arithmetic_v<T>) {
    // Overloads for integers have a return type of double.
    // We want to preserve the input type T.
    return t;
  } else {
    return real(t);
  }
}

// Inline static variables require C++17.
constexpr inline auto real_if_needed = [](const auto& t)
{
  using T = std::remove_const_t<decltype(t)>;
  return real_if_needed_impl(t, has_real<T>{});
};

} // end namespace impl
} // end namespace linalg
} // end inline namespace __p1673_version_0
} // end namespace MDSPAN_IMPL_PROPOSED_NAMESPACE
} // end namespace MDSPAN_IMPL_STANDARD_NAMESPACE

#endif //LINALG_INCLUDE_EXPERIMENTAL___P1673_BITS_REAL_IF_NEEDED_HPP_
