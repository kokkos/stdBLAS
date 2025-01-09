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

#ifndef LINALG_INCLUDE_EXPERIMENTAL___P1673_BITS_BLAS1_VECTOR_SUM_OF_SQUARES_HPP_
#define LINALG_INCLUDE_EXPERIMENTAL___P1673_BITS_BLAS1_VECTOR_SUM_OF_SQUARES_HPP_

#include <cmath>
#include <cstdlib>

namespace MDSPAN_IMPL_STANDARD_NAMESPACE {
namespace MDSPAN_IMPL_PROPOSED_NAMESPACE {
inline namespace __p1673_version_0 {
namespace linalg {

// Scaled sum of squares of a vector's elements
template<class Scalar>
struct sum_of_squares_result {
  Scalar scaling_factor;
  Scalar scaled_sum_of_squares;
};

namespace
{
template <class Exec, class x_t, class Scalar, class = void>
struct is_custom_vector_sum_of_squares_avail : std::false_type {};

template <class Exec, class x_t, class Scalar>
struct is_custom_vector_sum_of_squares_avail<
  Exec, x_t, Scalar,
  std::enable_if_t<
    std::is_same<
      decltype(vector_sum_of_squares(std::declval<Exec>(),
				     std::declval<x_t>(),
				     std::declval<sum_of_squares_result<Scalar>>()
				     )
	       ),
      sum_of_squares_result<Scalar>
      >::value
    && ! impl::is_inline_exec_v<Exec>
    >
  >
  : std::true_type{};

} // end anonymous namespace

template<class ElementType,
	 class SizeType,
         ::std::size_t ext0,
         class Layout,
         class Accessor,
         class Scalar>
sum_of_squares_result<Scalar> vector_sum_of_squares(
  impl::inline_exec_t&& /* exec */,
  mdspan<ElementType, extents<SizeType, ext0>, Layout, Accessor> x,
  sum_of_squares_result<Scalar> init)
{
  using std::abs;

  if (x.extent(0) == 0) {
    return init;
  }

  // Rescaling, as in the Reference BLAS DNRM2 implementation, avoids
  // unwarranted overflow or underflow.

  Scalar scale = init.scaling_factor;
  Scalar ssq = init.scaled_sum_of_squares;
  for (SizeType i = 0; i < x.extent(0); ++i) {
    if (abs(x(i)) != 0.0) {
      const auto absxi = abs(x(i));
      if (scale < absxi) {
          const auto quotient = scale / absxi;
          ssq = Scalar(1.0) + ssq * quotient * quotient;
          scale = absxi;
      }
      else {
        const auto quotient = absxi / scale;
        ssq = ssq + quotient * quotient;
      }
    }
  }

  sum_of_squares_result<Scalar> result;
  result.scaled_sum_of_squares = ssq;
  result.scaling_factor = scale;
  return result;
}

template<class ExecutionPolicy,
         class ElementType,
	 class SizeType,
         ::std::size_t ext0,
         class Layout,
         class Accessor,
         class Scalar>
sum_of_squares_result<Scalar> vector_sum_of_squares(
  ExecutionPolicy&& exec,
  mdspan<ElementType, extents<SizeType, ext0>, Layout, Accessor> v,
  sum_of_squares_result<Scalar> init)
{
  constexpr bool use_custom = is_custom_vector_sum_of_squares_avail<
    decltype(impl::map_execpolicy_with_check(exec)), decltype(v), Scalar
    >::value;

  if constexpr (use_custom) {
    return vector_sum_of_squares(impl::map_execpolicy_with_check(exec), v, init);
  }
  else {
    return vector_sum_of_squares(impl::inline_exec_t{}, v, init);
  }
}

template<class ElementType,
	 class SizeType,
         ::std::size_t ext0,
         class Layout,
         class Accessor,
         class Scalar>
sum_of_squares_result<Scalar> vector_sum_of_squares(
  mdspan<ElementType, extents<SizeType, ext0>, Layout, Accessor> v,
  sum_of_squares_result<Scalar> init)
{
  return vector_sum_of_squares(impl::default_exec_t{}, v, init);
}


} // end namespace linalg
} // end inline namespace __p1673_version_0
} // end namespace MDSPAN_IMPL_PROPOSED_NAMESPACE
} // end namespace MDSPAN_IMPL_STANDARD_NAMESPACE

#endif //LINALG_INCLUDE_EXPERIMENTAL___P1673_BITS_BLAS1_VECTOR_SUM_OF_SQUARES_HPP_
