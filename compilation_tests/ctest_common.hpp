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

#include <mdspan/mdspan.hpp>

#include <type_traits>

#pragma once

#define MDSPAN_STATIC_TEST(...) \
  static_assert(__VA_ARGS__, "MDSpan compile time test failed at "  __FILE__ ":" MDSPAN_PP_STRINGIFY(__LINE__))


// All tests need a main so that they'll link
int main() { }
