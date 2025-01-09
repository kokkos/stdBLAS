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

#ifndef LINALG_TPLIMPLEMENTATIONS_INCLUDE_EXPERIMENTAL___P1673_BITS_KOKKOSKERNELS_UTILS_HPP_
#define LINALG_TPLIMPLEMENTATIONS_INCLUDE_EXPERIMENTAL___P1673_BITS_KOKKOSKERNELS_UTILS_HPP_

#include <string_view>

namespace KokkosKernelsSTD {
namespace Impl {

#if defined(KOKKOS_STDBLAS_ENABLE_TESTS)
extern void signal_kokkos_impl_called(std::string_view functionName);
#else
void signal_kokkos_impl_called(std::string_view /* functionName */) {}
#endif

} // namespace Impl
} // namespace KokkosKernelsSTD
#endif
