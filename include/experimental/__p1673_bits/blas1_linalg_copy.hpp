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

#ifndef LINALG_INCLUDE_EXPERIMENTAL___P1673_BITS_BLAS1_LINALG_COPY_HPP_
#define LINALG_INCLUDE_EXPERIMENTAL___P1673_BITS_BLAS1_LINALG_COPY_HPP_

namespace MDSPAN_IMPL_STANDARD_NAMESPACE {
namespace MDSPAN_IMPL_PROPOSED_NAMESPACE {
inline namespace __p1673_version_0 {
namespace linalg {

namespace {

template<class ElementType_x,
	 class SizeType_x,
         ::std::size_t ext_x,
         class Layout_x,
         class Accessor_x,
         class ElementType_y,
	 class SizeType_y,
         ::std::size_t ext_y,
         class Layout_y,
         class Accessor_y>
void copy_rank_1(
  mdspan<ElementType_x, extents<SizeType_x, ext_x>, Layout_x, Accessor_x> x,
  mdspan<ElementType_y, extents<SizeType_y, ext_y>, Layout_y, Accessor_y> y)
{
  static_assert(x.static_extent(0) == dynamic_extent ||
                y.static_extent(0) == dynamic_extent ||
                x.static_extent(0) == y.static_extent(0));
  using size_type = std::common_type_t<SizeType_x, SizeType_y>;
  for (size_type i = 0; i < y.extent(0); ++i) {
    y(i) = x(i);
  }
}

template<class ElementType_x,
	 class SizeType_x,
         ::std::size_t numRows_x,
         ::std::size_t numCols_x,
         class Layout_x,
         class Accessor_x,
         class ElementType_y,
	 class SizeType_y,
         ::std::size_t numRows_y,
         ::std::size_t numCols_y,
         class Layout_y,
         class Accessor_y>
void copy_rank_2(
  mdspan<ElementType_x, extents<SizeType_x, numRows_x, numCols_x>, Layout_x, Accessor_x> x,
  mdspan<ElementType_y, extents<SizeType_y, numRows_y, numCols_y>, Layout_y, Accessor_y> y)
{
  static_assert(x.static_extent(0) == dynamic_extent ||
                y.static_extent(0) == dynamic_extent ||
                x.static_extent(0) == y.static_extent(0));
  static_assert(x.static_extent(1) == dynamic_extent ||
                y.static_extent(1) == dynamic_extent ||
                x.static_extent(1) == y.static_extent(1));
  using size_type = std::common_type_t<SizeType_x, SizeType_y>;
  for (size_type j = 0; j < y.extent(1); ++j) {
    for (size_type i = 0; i < y.extent(0); ++i) {
      y(i,j) = x(i,j);
    }
  }
}

template <class Exec, class x_t, class y_t, class = void>
struct is_custom_copy_avail : std::false_type {};

template <class Exec, class x_t, class y_t>
struct is_custom_copy_avail<
  Exec, x_t, y_t,
  std::enable_if_t<
    std::is_void_v<
      decltype(copy
	       (std::declval<Exec>(),
		std::declval<x_t>(),
		std::declval<y_t>()
		)
	       )
      >
    && ! impl::is_inline_exec_v<Exec>
    >
  >
  : std::true_type{};

} // end anonymous namespace


MDSPAN_TEMPLATE_REQUIRES(
         class ElementType_x,
	 class SizeType_x,
         ::std::size_t ... ext_x,
         class Layout_x,
         class Accessor_x,
         class ElementType_y,
	 class SizeType_y,
         ::std::size_t ... ext_y,
         class Layout_y,
         class Accessor_y,
         /* requires */ (sizeof...(ext_x) == sizeof...(ext_y) && sizeof...(ext_x) <= 2)
)
void copy(
  impl::inline_exec_t&& /* exec */,
  mdspan<ElementType_x, extents<SizeType_x, ext_x ...>, Layout_x, Accessor_x> x,
  mdspan<ElementType_y, extents<SizeType_y, ext_y ...>, Layout_y, Accessor_y> y)
{
  if constexpr (x.rank() == 1) {
    copy_rank_1(x, y);
  }
  else if constexpr (x.rank() == 2) {
    copy_rank_2(x, y);
  }
}

MDSPAN_TEMPLATE_REQUIRES(
         class ExecutionPolicy,
         class ElementType_x,
	 class SizeType_x,
         ::std::size_t ... ext_x,
         class Layout_x,
         class Accessor_x,
         class ElementType_y,
	 class SizeType_y,
         ::std::size_t ... ext_y,
         class Layout_y,
         class Accessor_y,
         /* requires */ (sizeof...(ext_x) == sizeof...(ext_y))
)
void copy(
  ExecutionPolicy&& exec,
  mdspan<ElementType_x, extents<SizeType_x, ext_x ...>, Layout_x, Accessor_x> x,
  mdspan<ElementType_y, extents<SizeType_y, ext_y ...>, Layout_y, Accessor_y> y)
{
  constexpr bool use_custom = is_custom_copy_avail<
    decltype(impl::map_execpolicy_with_check(exec)), decltype(x), decltype(y)
    >::value;

  if constexpr (use_custom) {
    copy(impl::map_execpolicy_with_check(exec), x, y);
  }
  else {
    copy(impl::inline_exec_t{}, x, y);
  }
}

MDSPAN_TEMPLATE_REQUIRES(
         class ElementType_x,
	 class SizeType_x,
         ::std::size_t ... ext_x,
         class Layout_x,
         class Accessor_x,
         class ElementType_y,
	 class SizeType_y,
         ::std::size_t ... ext_y,
         class Layout_y,
         class Accessor_y,
         /* requires */ (sizeof...(ext_x) == sizeof...(ext_y))
)
void copy(
  mdspan<ElementType_x, extents<SizeType_x, ext_x ...>, Layout_x, Accessor_x> x,
  mdspan<ElementType_y, extents<SizeType_y, ext_y ...>, Layout_y, Accessor_y> y)
{
  copy(impl::default_exec_t(), x, y);
}

} // end namespace linalg
} // end inline namespace __p1673_version_0
} // end namespace MDSPAN_IMPL_PROPOSED_NAMESPACE
} // end namespace MDSPAN_IMPL_STANDARD_NAMESPACE

#endif //LINALG_INCLUDE_EXPERIMENTAL___P1673_BITS_BLAS1_LINALG_COPY_HPP_
