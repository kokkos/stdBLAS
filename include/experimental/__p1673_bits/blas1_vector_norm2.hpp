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

#ifndef LINALG_INCLUDE_EXPERIMENTAL___P1673_BITS_BLAS1_VECTOR_NORM2_HPP_
#define LINALG_INCLUDE_EXPERIMENTAL___P1673_BITS_BLAS1_VECTOR_NORM2_HPP_

#include "blas1_vector_sum_of_squares.hpp"
#include <cmath>
#include <cstdlib>

namespace MDSPAN_IMPL_STANDARD_NAMESPACE {
namespace MDSPAN_IMPL_PROPOSED_NAMESPACE {
inline namespace __p1673_version_0 {
namespace linalg {

namespace
{
template <class Exec, class x_t, class Scalar, class = void>
struct is_custom_vector_two_norm_avail : std::false_type {};

template <class Exec, class x_t, class Scalar>
struct is_custom_vector_two_norm_avail<
  Exec, x_t, Scalar,
  std::enable_if_t<
    std::is_same<
      decltype(
	       vector_two_norm(std::declval<Exec>(),
			    std::declval<x_t>(),
			    std::declval<Scalar>())
	       ),
      Scalar
      >::value
    && ! impl::is_inline_exec_v<Exec>
    >
  >
  : std::true_type{};
} // end anonymous namespace

template<class ElementType,
         class SizeType, ::std::size_t ext0,
         class Layout,
         class Accessor,
         class Scalar>
Scalar vector_two_norm(
  impl::inline_exec_t&& exec,
  mdspan<ElementType, extents<SizeType, ext0>, Layout, Accessor> x,
  Scalar init)
{
  // Initialize the sum of squares result
  sum_of_squares_result<Scalar> ssq_init;
  ssq_init.scaling_factor = Scalar{};
  // FIXME (Hoemmen 2021/05/27) We'll need separate versions of this
  // for types whose "one" we don't know how to construct.
  ssq_init.scaled_sum_of_squares = 1.0;

  // Compute the sum of squares using an algorithm that avoids
  // underflow and overflow by scaling.
  auto ssq_res = vector_sum_of_squares(exec, x, ssq_init);
  using std::sqrt;
  return init + ssq_res.scaling_factor * sqrt(ssq_res.scaled_sum_of_squares);
}

template<class ExecutionPolicy,
         class ElementType,
         class SizeType, ::std::size_t ext0,
         class Layout,
         class Accessor,
         class Scalar>
Scalar vector_two_norm(
  ExecutionPolicy&& exec,
  mdspan<ElementType, extents<SizeType, ext0>, Layout, Accessor> x,
  Scalar init)
{
  constexpr bool use_custom = is_custom_vector_two_norm_avail<
    decltype(impl::map_execpolicy_with_check(exec)), decltype(x), Scalar
    >::value;

  if constexpr (use_custom) {
    return vector_two_norm(impl::map_execpolicy_with_check(exec), x, init);
  }
  else {
    return vector_two_norm(impl::inline_exec_t{}, x, init);
  }
}

template<class ElementType,
         class SizeType, ::std::size_t ext0,
         class Layout,
         class Accessor,
         class Scalar>
Scalar vector_two_norm(
  mdspan<ElementType, extents<SizeType, ext0>, Layout, Accessor> x,
  Scalar init)
{
  return vector_two_norm(impl::default_exec_t{}, x, init);
}


namespace vector_two_norm_detail {
  using std::abs;

  // The point of this is to do correct ADL for abs,
  // without exposing "using std::abs" in the outer namespace.
  template<
    class ElementType,
    class SizeType, ::std::size_t ext0,
    class Layout,
    class Accessor>
  auto vector_two_norm_return_type_deducer(
    mdspan<ElementType, extents<SizeType, ext0>, Layout, Accessor> x)
  -> decltype(abs(x(0)) * abs(x(0)));
} // namespace vector_two_norm_detail

template<class ElementType,
         class SizeType, ::std::size_t ext0,
         class Layout,
         class Accessor>
auto vector_two_norm(
  mdspan<ElementType, extents<SizeType, ext0>, Layout, Accessor> x)
-> decltype(vector_two_norm_detail::vector_two_norm_return_type_deducer(x))
{
  using return_t = decltype(vector_two_norm_detail::vector_two_norm_return_type_deducer(x));
  return vector_two_norm(x, return_t{});
}

template<class ExecutionPolicy,
         class ElementType,
         class SizeType, ::std::size_t ext0,
         class Layout,
         class Accessor>
auto vector_two_norm(
  ExecutionPolicy&& exec,
  mdspan<ElementType, extents<SizeType, ext0>, Layout, Accessor> x)
-> decltype(vector_two_norm_detail::vector_two_norm_return_type_deducer(x))
{
  using return_t = decltype(vector_two_norm_detail::vector_two_norm_return_type_deducer(x));
  return vector_two_norm(exec, x, return_t{});
}

} // end namespace linalg
} // end inline namespace __p1673_version_0
} // end namespace MDSPAN_IMPL_PROPOSED_NAMESPACE
} // end namespace MDSPAN_IMPL_STANDARD_NAMESPACE

#endif //LINALG_INCLUDE_EXPERIMENTAL___P1673_BITS_BLAS1_VECTOR_NORM2_HPP_
