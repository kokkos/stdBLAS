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

#ifndef LINALG_INCLUDE_EXPERIMENTAL___P1673_BITS_CONJUGATE_IF_NEEDED_HPP_
#define LINALG_INCLUDE_EXPERIMENTAL___P1673_BITS_CONJUGATE_IF_NEEDED_HPP_

#include <complex>
#include <type_traits>

namespace MDSPAN_IMPL_STANDARD_NAMESPACE {
namespace MDSPAN_IMPL_PROPOSED_NAMESPACE {
inline namespace __p1673_version_0 {
namespace linalg {
namespace impl {

template<class T> struct is_complex : std::false_type{};

template<> struct is_complex<std::complex<float>> : std::true_type{};
template<> struct is_complex<std::complex<double>> : std::true_type{};
template<> struct is_complex<std::complex<long double>> : std::true_type{};

template<class T> inline constexpr bool is_complex_v = is_complex<T>::value;

template<class T, class = void>
struct has_conj : std::false_type {};

// If I can find unqualified conj via overload resolution,
// then assume that conj(t) returns the conjugate of t.
template<class T>
struct has_conj<T, decltype(conj(std::declval<T>()), void())> : std::true_type {};

template<class T>
T conj_if_needed_impl(const T& t, std::false_type)
{
  return t;
}

template<class T>
auto conj_if_needed_impl(const T& t, std::true_type)
{
  if constexpr (std::is_arithmetic_v<T>) {
    return t;
  } else {
    return conj(t);
  }
}

// Inline static variables require C++17.
constexpr inline auto conj_if_needed = [](const auto& t)
{
  using T = std::remove_const_t<decltype(t)>;
  return conj_if_needed_impl(t, has_conj<T>{});
};

} // end namespace impl
} // end namespace linalg
} // end inline namespace __p1673_version_0
} // end namespace MDSPAN_IMPL_PROPOSED_NAMESPACE
} // end namespace MDSPAN_IMPL_STANDARD_NAMESPACE

#endif //LINALG_INCLUDE_EXPERIMENTAL___P1673_BITS_CONJUGATE_IF_NEEDED_HPP_
