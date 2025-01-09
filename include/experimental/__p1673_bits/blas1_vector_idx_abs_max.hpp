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

#ifndef LINALG_INCLUDE_EXPERIMENTAL___P1673_BITS_BLAS1_VECTOR_IDX_ABS_MAX_HPP_
#define LINALG_INCLUDE_EXPERIMENTAL___P1673_BITS_BLAS1_VECTOR_IDX_ABS_MAX_HPP_

namespace MDSPAN_IMPL_STANDARD_NAMESPACE {
namespace MDSPAN_IMPL_PROPOSED_NAMESPACE {
inline namespace __p1673_version_0 {
namespace linalg {

// begin anonymous namespace
namespace {

template <class Exec, class v_t, class = void>
struct is_custom_vector_idx_abs_max_avail : std::false_type {};

template <class Exec, class v_t>
struct is_custom_vector_idx_abs_max_avail<
  Exec, v_t,
  std::enable_if_t<
    //FRizzi: maybe should use is_convertible?
    std::is_same<
      decltype(vector_idx_abs_max(std::declval<Exec>(), std::declval<v_t>())),
      typename v_t::extents_type::size_type
      >::value
    && ! impl::is_inline_exec_v<Exec>
    >
  >
  : std::true_type{};

template<class ElementType,
         class SizeType, ::std::size_t ext0,
         class Layout,
         class Accessor>
SizeType vector_idx_abs_max_default_impl(
  mdspan<ElementType, extents<SizeType, ext0>, Layout, Accessor> v)
{
  using std::abs;
  using value_type = typename decltype(v)::value_type;
  using magnitude_type =
    decltype(impl::abs_if_needed(impl::real_if_needed(std::declval<value_type>())) +
             impl::abs_if_needed(impl::imag_if_needed(std::declval<value_type>())));

  if (v.extent(0) == 0) {
    return std::numeric_limits<SizeType>::max();
  }

  if constexpr (std::is_arithmetic_v<value_type>) {
    SizeType maxInd = 0;
    magnitude_type maxVal = abs(v(0));
    for (SizeType i = 1; i < v.extent(0); ++i) {
      if (maxVal < abs(v(i))) {
        maxVal = abs(v(i));
        maxInd = i;
      }
    }

    return maxInd;
  }
  else {
    SizeType maxInd = 0;
    magnitude_type maxVal = impl::abs_if_needed(impl::real_if_needed(v(0))) +
                            impl::abs_if_needed(impl::imag_if_needed(v(0)));

    for (SizeType i = 1; i < v.extent(0); ++i) {
      magnitude_type val = impl::abs_if_needed(impl::real_if_needed(v(i))) +
                           impl::abs_if_needed(impl::imag_if_needed(v(i)));

      if (maxVal < val) {
        maxVal = val;
        maxInd = i;
      }
    }

    return maxInd;
  }
}

} // end anonymous namespace

template<class ElementType,
         class SizeType, ::std::size_t ext0,
         class Layout,
         class Accessor>
SizeType vector_idx_abs_max(
  impl::inline_exec_t&& /* exec */,
  mdspan<ElementType, extents<SizeType, ext0>, Layout, Accessor> v)
{
  return vector_idx_abs_max_default_impl(v);
}

template<class ExecutionPolicy,
         class ElementType,
         class SizeType, ::std::size_t ext0,
         class Layout,
         class Accessor>
SizeType vector_idx_abs_max(
  ExecutionPolicy&& exec,
  mdspan<ElementType, extents<SizeType, ext0>, Layout, Accessor> v)
{
  if (v.extent(0) == 0) {
    return std::numeric_limits<SizeType>::max();
  }

  constexpr bool use_custom = is_custom_vector_idx_abs_max_avail<
    decltype(impl::map_execpolicy_with_check(exec)), decltype(v)
    >::value;

  if constexpr (use_custom) {
    return vector_idx_abs_max(impl::map_execpolicy_with_check(exec), v);
  }
  else {
    return vector_idx_abs_max(impl::inline_exec_t{}, v);
  }
}

template<class ElementType,
         class SizeType, ::std::size_t ext0,
         class Layout,
         class Accessor>
SizeType vector_idx_abs_max(
  mdspan<ElementType, extents<SizeType, ext0>, Layout, Accessor> v)
{
  return vector_idx_abs_max(impl::default_exec_t{}, v);
}

} // end namespace linalg
} // end inline namespace __p1673_version_0
} // end namespace MDSPAN_IMPL_PROPOSED_NAMESPACE
} // end namespace MDSPAN_IMPL_STANDARD_NAMESPACE

#endif //LINALG_INCLUDE_EXPERIMENTAL___P1673_BITS_BLAS1_VECTOR_IDX_ABS_MAX_HPP_
