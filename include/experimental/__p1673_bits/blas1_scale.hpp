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

#ifndef LINALG_INCLUDE_EXPERIMENTAL___P1673_BITS_BLAS1_SCALE_HPP_
#define LINALG_INCLUDE_EXPERIMENTAL___P1673_BITS_BLAS1_SCALE_HPP_

namespace MDSPAN_IMPL_STANDARD_NAMESPACE {
namespace MDSPAN_IMPL_PROPOSED_NAMESPACE {
inline namespace __p1673_version_0 {
namespace linalg {

namespace {

template<class ElementType,
	 class SizeType,
         ::std::size_t ext0,
         class Layout,
         class Accessor,
         class Scalar>
void linalg_scale_rank_1(
  const Scalar alpha,
  mdspan<ElementType, extents<SizeType, ext0>, Layout, Accessor> x)
{
  for (SizeType i = 0; i < x.extent(0); ++i) {
    x(i) *= alpha;
  }
}

template<class ElementType,
	 class SizeType,
         ::std::size_t numRows,
         ::std::size_t numCols,
         class Layout,
         class Accessor,
         class Scalar>
void linalg_scale_rank_2(
  const Scalar alpha,
  mdspan<ElementType, extents<SizeType, numRows, numCols>, Layout, Accessor> A)
{
  for (SizeType j = 0; j < A.extent(1); ++j) {
    for (SizeType i = 0; i < A.extent(0); ++i) {
      A(i,j) *= alpha;
    }
  }
}

template <typename Exec, typename Scalar, typename x_t, typename = void>
struct is_custom_scale_avail : std::false_type {};

template <typename Exec, typename Scalar, typename x_t>
struct is_custom_scale_avail<
  Exec, Scalar, x_t,
  std::enable_if_t<
    std::is_void_v<
      decltype(scale(std::declval<Exec>(),
		     std::declval<Scalar>(),
		     std::declval<x_t>()))
      >
    && ! impl::is_inline_exec_v<Exec>
    >
  >
  : std::true_type{};

} // end anonymous namespace

template<class Scalar,
         class ElementType,
	 class SizeType,
         ::std::size_t ... ext,
         class Layout,
         class Accessor>
void scale(
  impl::inline_exec_t&& /* exec */,
  const Scalar alpha,
  mdspan<ElementType, extents<SizeType, ext ...>, Layout, Accessor> x)
{
  static_assert(x.rank() <= 2);

  if constexpr (x.rank() == 1) {
    linalg_scale_rank_1(alpha, x);
  }
  else if constexpr (x.rank() == 2) {
    linalg_scale_rank_2(alpha, x);
  }
}

template<class ExecutionPolicy,
         class Scalar,
         class ElementType,
	 class SizeType,
         ::std::size_t ... ext,
         class Layout,
         class Accessor>
void scale(
  ExecutionPolicy&& exec,
  const Scalar alpha,
  mdspan<ElementType, extents<SizeType, ext ...>, Layout, Accessor> x)
{
  // Call custom overload if available else call std implementation

  constexpr bool use_custom = is_custom_scale_avail<
    decltype(impl::map_execpolicy_with_check(exec)), decltype(alpha), decltype(x)
    >::value;

  if constexpr (use_custom) {
    scale(impl::map_execpolicy_with_check(exec), alpha, x);
  } else {
    scale(impl::inline_exec_t{}, alpha, x);
  }
}

template<class Scalar,
         class ElementType,
	 class SizeType,
         ::std::size_t ... ext,
         class Layout,
         class Accessor>
void scale(const Scalar alpha,
           mdspan<ElementType, extents<SizeType, ext ...>, Layout, Accessor> x)
{
  scale(impl::default_exec_t{}, alpha, x);
}

} // end namespace linalg
} // end inline namespace __p1673_version_0
} // end namespace MDSPAN_IMPL_PROPOSED_NAMESPACE
} // end namespace MDSPAN_IMPL_STANDARD_NAMESPACE

#endif //LINALG_INCLUDE_EXPERIMENTAL___P1673_BITS_BLAS1_SCALE_HPP_
