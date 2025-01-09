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

#ifndef LINALG_INCLUDE_EXPERIMENTAL___P1673_BITS_BLAS1_DOT_HPP_
#define LINALG_INCLUDE_EXPERIMENTAL___P1673_BITS_BLAS1_DOT_HPP_

#include <type_traits>

namespace MDSPAN_IMPL_STANDARD_NAMESPACE {
namespace MDSPAN_IMPL_PROPOSED_NAMESPACE {
inline namespace __p1673_version_0 {
namespace linalg {

// begin anonymous namespace
namespace {

template <class Exec, class v1_t, class v2_t, class Scalar, class = void>
struct is_custom_dot_avail : std::false_type {};

template <class Exec, class v1_t, class v2_t, class Scalar>
struct is_custom_dot_avail<
  Exec, v1_t, v2_t, Scalar,
  std::enable_if_t<
    std::is_same<
      decltype(dot
	       (std::declval<Exec>(),
		std::declval<v1_t>(),
		std::declval<v2_t>(),
		std::declval<Scalar>()
		)
	       ),
      Scalar
      >::value
    && ! impl::is_inline_exec_v<Exec>
    >
  >
  : std::true_type {};

} // end anonymous namespace


template<class ElementType1,
	 class SizeType1,
         ::std::size_t ext1,
         class Layout1,
         class Accessor1,
         class ElementType2,
	 class SizeType2,	 
         ::std::size_t ext2,
         class Layout2,
         class Accessor2,
         class Scalar>
Scalar dot(
  impl::inline_exec_t&& /* exec */,
  mdspan<ElementType1, extents<SizeType1, ext1>, Layout1, Accessor1> v1,
  mdspan<ElementType2, extents<SizeType2, ext2>, Layout2, Accessor2> v2,
  Scalar init)
{
  static_assert(v1.static_extent(0) == dynamic_extent ||
                v2.static_extent(0) == dynamic_extent ||
                v1.static_extent(0) == v2.static_extent(0));

  using size_type = std::common_type_t<SizeType1, SizeType2>;
  for (size_type k = 0; k < v1.extent(0); ++k) {
    init += v1(k) * v2(k);
  }
  return init;
}

template<class ExecutionPolicy,
         class ElementType1,
	 class SizeType1,	 
         ::std::size_t ext1,
         class Layout1,
         class Accessor1,
         class ElementType2,
	 class SizeType2,
         ::std::size_t ext2,
         class Layout2,
         class Accessor2,
         class Scalar>
Scalar dot(
  ExecutionPolicy&& exec ,
  mdspan<ElementType1, extents<SizeType1, ext1>, Layout1, Accessor1> v1,
  mdspan<ElementType2, extents<SizeType2, ext2>, Layout2, Accessor2> v2,
  Scalar init)
{
  static_assert(v1.static_extent(0) == dynamic_extent ||
                v2.static_extent(0) == dynamic_extent ||
                v1.static_extent(0) == v2.static_extent(0));

  constexpr bool use_custom = is_custom_dot_avail<
    decltype(impl::map_execpolicy_with_check(exec)), decltype(v1), decltype(v2), Scalar
    >::value;

  if constexpr (use_custom) {
    return dot(impl::map_execpolicy_with_check(exec), v1, v2, init);
  }
  else {
    return dot(impl::inline_exec_t{}, v1, v2, init);
  }
}

template<class ElementType1,
	 class SizeType1,
         ::std::size_t ext1,
         class Layout1,
         class Accessor1,
         class ElementType2,
	 class SizeType2,
         ::std::size_t ext2,
         class Layout2,
         class Accessor2,
         class Scalar>
Scalar dot(mdspan<ElementType1, extents<SizeType1, ext1>, Layout1, Accessor1> v1,
           mdspan<ElementType2, extents<SizeType2, ext2>, Layout2, Accessor2> v2,
           Scalar init)
{
  return dot(impl::default_exec_t{}, v1, v2, init);
}

template<class ElementType1,
	 class SizeType1,
         ::std::size_t ext1,
         class Layout1,
         class Accessor1,
         class ElementType2,
	 class SizeType2,
         ::std::size_t ext2,
         class Layout2,
         class Accessor2,
         class Scalar>
Scalar dotc(
  mdspan<ElementType1, extents<SizeType1, ext1>, Layout1, Accessor1> v1,
  mdspan<ElementType2, extents<SizeType2, ext2>, Layout2, Accessor2> v2,
  Scalar init)
{
  return dot(conjugated(v1), v2, init);
}

template<class ExecutionPolicy,
         class ElementType1,
	 class SizeType1,
         ::std::size_t ext1,
         class Layout1,
         class Accessor1,
         class ElementType2,
	 class SizeType2,
         ::std::size_t ext2,
         class Layout2,
         class Accessor2,
         class Scalar>
Scalar dotc(
  ExecutionPolicy&& exec,
  mdspan<ElementType1, extents<SizeType1, ext1>, Layout1, Accessor1> v1,
  mdspan<ElementType2, extents<SizeType2, ext2>, Layout2, Accessor2> v2,
  Scalar init)
{
  return dot(exec, conjugated(v1), v2, init);
}

namespace dot_detail {
  using std::abs;

  // The point of this is to do correct ADL for abs,
  // without exposing "using std::abs" in the outer namespace.
  template<
    class ElementType1,
    class SizeType1,
    ::std::size_t ext1,
    class Layout1,
    class Accessor1,
    class ElementType2,
    class SizeType2,
    ::std::size_t ext2,
    class Layout2,
    class Accessor2>
  auto dot_return_type_deducer(
    mdspan<ElementType1, extents<SizeType1, ext1>, Layout1, Accessor1> x,
    mdspan<ElementType2, extents<SizeType2, ext2>, Layout2, Accessor2> y)
  -> decltype(x(0) * y(0));
} // namespace dot_detail


template<class ElementType1,
	 class SizeType1,
         ::std::size_t ext1,
         class Layout1,
         class Accessor1,
         class ElementType2,
	 class SizeType2,
         ::std::size_t ext2,
         class Layout2,
         class Accessor2>
auto dot(
  mdspan<ElementType1, extents<SizeType1, ext1>, Layout1, Accessor1> v1,
  mdspan<ElementType2, extents<SizeType2, ext2>, Layout2, Accessor2> v2)
-> decltype(dot_detail::dot_return_type_deducer(v1, v2))
{
  using return_t = decltype(dot_detail::dot_return_type_deducer(v1, v2));
  return dot(v1, v2, return_t{});
}

template<class ExecutionPolicy,
         class ElementType1,
	 class SizeType1,
         ::std::size_t ext1,
         class Layout1,
         class Accessor1,
         class ElementType2,
	 class SizeType2,
         ::std::size_t ext2,
         class Layout2,
         class Accessor2>
auto dot(
  ExecutionPolicy&& exec,
  mdspan<ElementType1, extents<SizeType1, ext1>, Layout1, Accessor1> v1,
  mdspan<ElementType2, extents<SizeType2, ext2>, Layout2, Accessor2> v2)
-> decltype(dot_detail::dot_return_type_deducer(v1, v2))
{
  using return_t = decltype(dot_detail::dot_return_type_deducer(v1, v2));
  return dot(exec, v1, v2, return_t{});
}

template<class ElementType1,
	 class SizeType1,
         ::std::size_t ext1,
         class Layout1,
         class Accessor1,
         class ElementType2,
	 class SizeType2,
         ::std::size_t ext2,
         class Layout2,
         class Accessor2>
auto dotc(
  mdspan<ElementType1, extents<SizeType1, ext1>, Layout1, Accessor1> v1,
  mdspan<ElementType2, extents<SizeType2, ext2>, Layout2, Accessor2> v2)
  -> decltype(dot_detail::dot_return_type_deducer(conjugated(v1), v2))
{
  using return_t = decltype(dot_detail::dot_return_type_deducer(conjugated(v1), v2));
  return dotc(v1, v2, return_t{});
}

template<class ExecutionPolicy,
         class ElementType1,
	 class SizeType1,
         ::std::size_t ext1,
         class Layout1,
         class Accessor1,
         class ElementType2,
	 class SizeType2,
         ::std::size_t ext2,
         class Layout2,
         class Accessor2>
auto dotc(
  ExecutionPolicy&& exec,
  mdspan<ElementType1, extents<SizeType1, ext1>, Layout1, Accessor1> v1,
  mdspan<ElementType2, extents<SizeType2, ext2>, Layout2, Accessor2> v2)
 -> decltype(dot_detail::dot_return_type_deducer(conjugated(v1), v2))
{
  using return_t = decltype(dot_detail::dot_return_type_deducer(conjugated(v1), v2));
  return dotc(exec, v1, v2, return_t{});
}

} // end namespace linalg
} // end inline namespace __p1673_version_0
} // end namespace MDSPAN_IMPL_PROPOSED_NAMESPACE
} // end namespace MDSPAN_IMPL_STANDARD_NAMESPACE

#endif //LINALG_INCLUDE_EXPERIMENTAL___P1673_BITS_BLAS1_DOT_HPP_
