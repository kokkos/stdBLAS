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

#ifndef LINALG_TPLIMPLEMENTATIONS_INCLUDE_EXPERIMENTAL___P1673_BITS_KOKKOSKERNELS_CONJUGATE_IF_NEEDED_HPP_
#define LINALG_TPLIMPLEMENTATIONS_INCLUDE_EXPERIMENTAL___P1673_BITS_KOKKOSKERNELS_CONJUGATE_IF_NEEDED_HPP_

#include "experimental/__p1673_bits/conj_if_needed.hpp"

namespace std {
namespace experimental {
namespace linalg {
namespace impl{

// conj_if_needed doesn't use an is_complex trait.
// Instead, it checks whether conj(x) (namespace-unqualified) is a valid expression,
// calls that if so, else assumes that x represents a real number and returns x.
// Thus, we don't actually need to do anything here.

} // end namespace impl
} // end namespace linalg
} // end namespace experimental
} // end namespace std

#endif //LINALG_TPLIMPLEMENTATIONS_INCLUDE_EXPERIMENTAL___P1673_BITS_KOKKOSKERNELS_CONJUGATE_IF_NEEDED_HPP_
