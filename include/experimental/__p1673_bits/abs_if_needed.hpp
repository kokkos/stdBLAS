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

#ifndef LINALG_INCLUDE_EXPERIMENTAL___P1673_BITS_ABS_IF_NEEDED_HPP_
#define LINALG_INCLUDE_EXPERIMENTAL___P1673_BITS_ABS_IF_NEEDED_HPP_

#include <cmath>
#include <complex>
#include <type_traits>

namespace MDSPAN_IMPL_STANDARD_NAMESPACE {
namespace MDSPAN_IMPL_PROPOSED_NAMESPACE {
inline namespace __p1673_version_0 {
namespace linalg {
namespace impl {

// E if T is an unsigned integer;
//
// (1.2) otherwise, std::abs(E) if T is an arithmetic type,
//
// (1.3) otherwise, abs(E), if that expression is valid, with overload
//   resolution performed in a context that includes the declaration
//   template<class T> T abs(T) = delete;. If the function selected by
//   overload resolution does not return the absolute value of its
//   input, the program is ill-formed, no diagnostic required.

// Inline static variables require C++17.
constexpr inline auto abs_if_needed = [](auto t)
{
  using T = std::remove_const_t<std::remove_reference_t<decltype(t)>>;
  if constexpr (std::is_arithmetic_v<T>) {
    if constexpr (std::is_unsigned_v<T>) {
      return t;
    }
    else {
      return std::abs(t);
    }
  }
  else {
    return abs(t);
  }
};

} // end namespace impl
} // end namespace linalg
} // end inline namespace __p1673_version_0
} // end namespace MDSPAN_IMPL_PROPOSED_NAMESPACE
} // end namespace MDSPAN_IMPL_STANDARD_NAMESPACE

#endif //LINALG_INCLUDE_EXPERIMENTAL___P1673_BITS_ABS_IF_NEEDED_HPP_
