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

#ifndef LINALG_INCLUDE_EXPERIMENTAL___P1673_BITS_PACKED_LAYOUT_HPP_
#define LINALG_INCLUDE_EXPERIMENTAL___P1673_BITS_PACKED_LAYOUT_HPP_

#include <mdspan/mdspan.hpp>
#include "layout_triangle.hpp"

namespace MDSPAN_IMPL_STANDARD_NAMESPACE {
namespace MDSPAN_IMPL_PROPOSED_NAMESPACE {
inline namespace __p1673_version_0 {
namespace linalg {

// TODO declarations need extents-see-returns-below defined

#if 0
template<class EltType,
         class Extents,
         class Layout,
         class Accessor,
         class Triangle,
         class StorageOrder>
constexpr mdspan<EltType,
  <i>extents-see-returns-below</i>,
  layout_blas_packed<
    Triangle,
    StorageOrder>,
  Accessor>
packed(
  const mdspan<EltType, Extents, Layout, Accessor>& m,
  typename mdspan<EltType, Extents, Layout, Accessor>::index_type num_rows,
  Triangle,
  StorageOrder);
#endif // 0

} // end namespace linalg
} // end inline namespace __p1673_version_0
} // end namespace MDSPAN_IMPL_PROPOSED_NAMESPACE
} // end namespace MDSPAN_IMPL_STANDARD_NAMESPACE

#endif //LINALG_INCLUDE_EXPERIMENTAL___P1673_BITS_PACKED_LAYOUT_HPP_
