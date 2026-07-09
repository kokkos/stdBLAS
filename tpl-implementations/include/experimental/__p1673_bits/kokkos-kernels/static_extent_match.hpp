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

#ifndef LINALG_TPLIMPLEMENTATIONS_INCLUDE_EXPERIMENTAL___P1673_BITS_KOKKOSKERNELS_STATICEXTMATCH_HPP_
#define LINALG_TPLIMPLEMENTATIONS_INCLUDE_EXPERIMENTAL___P1673_BITS_KOKKOSKERNELS_STATICEXTMATCH_HPP_

namespace KokkosKernelsSTD {
namespace Impl {

template <class size_type>
constexpr bool static_extent_match(size_type extent1, size_type extent2)
{
  using MDSPAN_IMPL_STANDARD_NAMESPACE::dynamic_extent;

  return extent1 == dynamic_extent ||
         extent2 == dynamic_extent ||
         extent1 == extent2;
}

} // namespace Impl
} // namespace KokkosKernelsSTD
#endif
