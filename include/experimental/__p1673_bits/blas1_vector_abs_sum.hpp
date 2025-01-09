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

#ifndef LINALG_INCLUDE_EXPERIMENTAL___P1673_BITS_BLAS1_VECTOR_ABS_SUM_HPP_
#define LINALG_INCLUDE_EXPERIMENTAL___P1673_BITS_BLAS1_VECTOR_ABS_SUM_HPP_

#include <cstdlib>
#include <cmath>

namespace MDSPAN_IMPL_STANDARD_NAMESPACE {
namespace MDSPAN_IMPL_PROPOSED_NAMESPACE {
inline namespace __p1673_version_0 {
namespace linalg {

namespace
{
template <class Exec, class v_t, class Scalar, class = void>
struct is_custom_vector_abs_sum_avail : std::false_type {};

template <class Exec, class v_t, class Scalar>
struct is_custom_vector_abs_sum_avail<
  Exec, v_t, Scalar,
  std::enable_if_t<
    std::is_same<
      decltype(vector_abs_sum(std::declval<Exec>(),
			      std::declval<v_t>(),
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
Scalar vector_abs_sum(
  impl::inline_exec_t&& /* exec */,
  mdspan<ElementType, extents<SizeType, ext0>, Layout, Accessor> v,
  Scalar init)
{
  using value_type = typename decltype(v)::value_type;
  using sum_type =
    decltype(init +
             impl::abs_if_needed(impl::real_if_needed(std::declval<value_type>())) +
             impl::abs_if_needed(impl::imag_if_needed(std::declval<value_type>())));
  static_assert(std::is_convertible_v<sum_type, Scalar>);
  
  const SizeType numElt = v.extent(0);
  if constexpr (std::is_arithmetic_v<value_type>) {
    for (SizeType i = 0; i < numElt; ++i) {
      init += impl::abs_if_needed(v(i));
    }
  }
  else {
    for (SizeType i = 0; i < numElt; ++i) {
      init += impl::abs_if_needed(impl::real_if_needed(v(i)));
      init += impl::abs_if_needed(impl::imag_if_needed(v(i)));
    }
  }

  return init;
}

template<class ExecutionPolicy,
         class ElementType,
         class SizeType, ::std::size_t ext0,
         class Layout,
         class Accessor,
         class Scalar>
Scalar vector_abs_sum(
  ExecutionPolicy&& exec,
  mdspan<ElementType, extents<SizeType, ext0>, Layout, Accessor> v,
  Scalar init)
{
  constexpr bool use_custom = is_custom_vector_abs_sum_avail<
    decltype(impl::map_execpolicy_with_check(exec)), decltype(v), Scalar
    >::value;

  if constexpr (use_custom) {
    return vector_abs_sum(impl::map_execpolicy_with_check(exec), v, init);
  }
  else {
    return vector_abs_sum(impl::inline_exec_t{}, v, init);
  }
}

template<class ElementType,
         class SizeType, ::std::size_t ext0,
         class Layout,
         class Accessor,
         class Scalar>
Scalar vector_abs_sum(
  mdspan<ElementType, extents<SizeType, ext0>, Layout, Accessor> v,
  Scalar init)
{
  return vector_abs_sum(impl::default_exec_t{}, v, init);
}

namespace vector_abs_detail {
  using std::abs;

  // The point of this is to do correct ADL for abs,
  // without exposing "using std::abs" in the outer namespace.
  template<
    class ElementType,
    class SizeType, ::std::size_t ext0,
    class Layout,
    class Accessor>
  auto vector_abs_return_type_deducer(
    mdspan<ElementType, extents<SizeType, ext0>, Layout, Accessor> x)
  -> decltype(abs(x(0)));
} // namespace vector_abs_detail


template<class ElementType,
         class SizeType, ::std::size_t ext0,
         class Layout,
         class Accessor>
auto vector_abs_sum(
  mdspan<ElementType, extents<SizeType, ext0>, Layout, Accessor> x)
-> decltype(vector_abs_detail::vector_abs_return_type_deducer(x))
{
  using return_t = decltype(vector_abs_detail::vector_abs_return_type_deducer(x));
  return vector_abs_sum(x, return_t{});
}

template<class ExecutionPolicy,
         class ElementType,
         class SizeType, ::std::size_t ext0,
         class Layout,
         class Accessor>
auto vector_abs_sum(
  ExecutionPolicy&& exec,
  mdspan<ElementType, extents<SizeType, ext0>, Layout, Accessor> x)
-> decltype(vector_abs_detail::vector_abs_return_type_deducer(x))
{
  using return_t = decltype(vector_abs_detail::vector_abs_return_type_deducer(x));
  return vector_abs_sum(exec, x, return_t{});
}

} // end namespace linalg
} // end inline namespace __p1673_version_0
} // end namespace MDSPAN_IMPL_PROPOSED_NAMESPACE
} // end namespace MDSPAN_IMPL_STANDARD_NAMESPACE

#endif //LINALG_INCLUDE_EXPERIMENTAL___P1673_BITS_BLAS1_VECTOR_ABS_SUM_HPP_
