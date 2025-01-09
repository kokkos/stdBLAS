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

#ifndef LINALG_INCLUDE_EXPERIMENTAL___P1673_BITS_LAYOUT_TAGS_HPP_
#define LINALG_INCLUDE_EXPERIMENTAL___P1673_BITS_LAYOUT_TAGS_HPP_

#include <mdspan/mdspan.hpp>

namespace MDSPAN_IMPL_STANDARD_NAMESPACE {
namespace MDSPAN_IMPL_PROPOSED_NAMESPACE {
inline namespace __p1673_version_0 {
namespace linalg {

// TODO @proposal-bug make sure these can't convert from `{}`

struct column_major_t { };
_MDSPAN_INLINE_VARIABLE constexpr auto column_major = column_major_t{};
struct row_major_t { };
_MDSPAN_INLINE_VARIABLE constexpr auto row_major = row_major_t{};

struct upper_triangle_t { };
_MDSPAN_INLINE_VARIABLE constexpr auto upper_triangle = upper_triangle_t{};
struct lower_triangle_t { };
_MDSPAN_INLINE_VARIABLE constexpr auto lower_triangle = lower_triangle_t{};

struct implicit_unit_diagonal_t { };
_MDSPAN_INLINE_VARIABLE constexpr auto implicit_unit_diagonal = implicit_unit_diagonal_t{};
struct explicit_diagonal_t { };
_MDSPAN_INLINE_VARIABLE constexpr auto explicit_diagonal = explicit_diagonal_t{};

struct left_side_t { };
_MDSPAN_INLINE_VARIABLE constexpr auto left_side = left_side_t{};
struct right_side_t { };
_MDSPAN_INLINE_VARIABLE constexpr auto right_side = right_side_t{};

} // end namespace linalg
} // end inline namespace __p1673_version_0
} // end namespace MDSPAN_IMPL_PROPOSED_NAMESPACE
} // end namespace MDSPAN_IMPL_STANDARD_NAMESPACE


#endif //LINALG_INCLUDE_EXPERIMENTAL___P1673_BITS_LAYOUT_TAGS_HPP_
