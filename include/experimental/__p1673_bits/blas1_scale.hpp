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

#include "blas_helpers.hpp"
#include <type_traits>

namespace MDSPAN_IMPL_STANDARD_NAMESPACE {
namespace MDSPAN_IMPL_PROPOSED_NAMESPACE {
inline namespace __p1673_version_0 {
namespace linalg {

namespace {

template<class ElementType,
	 class IndexType,
         ::std::size_t ext0,
         class Layout,
         class Accessor,
         class Scalar>
void generic_scale_rank_1(
  const Scalar alpha,
  mdspan<ElementType, extents<IndexType, ext0>, Layout, Accessor> x)
{
  for (IndexType i = 0; i < x.extent(0); ++i) {
    x(i) *= alpha;
  }
}

template<class Scalar, class MdspanType>
constexpr bool maybe_can_blas_scale() {
  // It's OK if Scalar and value_type aren't the same,
  // as long as we can convert Scalar to value_type.
  using value_type = typename MdspanType::value_type;
  constexpr bool blas_value_type =
    std::is_convertible_v<Scalar, value_type> &&
    impl::is_blas_value_type_v<value_type>;

  constexpr bool blas_layout =
    impl::is_blas_layout_type_v<typename MdspanType::layout_type>;

  constexpr bool blas_accessor =
    impl::is_blas_accessor_type_v<typename MdspanType::accessor_type>;

  return blas_value_type && blas_layout && blas_accessor;
} 

// Return true if the BLAS call was successfull, false otherwise.
template<class ElementType,
	 class IndexType,
         ::std::size_t ext0,
         class Layout,
         class Accessor,
         class Scalar>
bool try_blas_scale(
  const Scalar alpha,
  mdspan<ElementType, extents<IndexType, ext0>, Layout, Accessor> x)
{
  #ifdef KOKKOS_ENABLE_CBLAS
  auto n = x.extent(0);
  auto incx = x.stride(0);
  if (x.is_strided() && n <= std::numeric_limits<int>::max()) {
    if constexpr (std::is_same_v<ElementType, float>) {
      cblas_sscal(n, alpha, x.data_handle(), incx);
      return true;
    } else if constexpr (std::is_same_v<ElementType, double>) {
      cblas_dscal(n, alpha, x.data_handle(), incx);
      return true;
    } else if constexpr (std::is_same_v<ElementType, std::complex<float>>) {
      if constexpr(std::is_convertible_v<Scalar, float>) {
        cblas_csscal(n, alpha, x.data_handle(), incx);
        return true;
      }
      else if constexpr (std::is_convertible_v<Scalar, std::complex<float>>) {
        auto converted_alpha = static_cast<std::complex<float>>(alpha);
        cblas_cscal(n, &converted_alpha, x.data_handle(), incx);
        return true;
      }
    } else if constexpr (std::is_same_v<ElementType, std::complex<double>>) {
      if constexpr (std::is_convertible_v<Scalar, double>) {
        cblas_zdscal(n, alpha, x.data_handle(), incx);
        return true;
      }
      else if constexpr (std::is_convertible_v<Scalar, std::complex<double>>) {
        auto converted_alpha = static_cast<std::complex<double>>(alpha);
        cblas_zscal(n, &converted_alpha, x.data_handle(), incx);
        return true;
      }
    }
  }
  #endif
  return false;
}

template<class ElementType,
	 class IndexType,
         ::std::size_t ext0,
         class Layout,
         class Accessor,
         class Scalar>
void linalg_scale_rank_1(
  const Scalar alpha,
  mdspan<ElementType, extents<IndexType, ext0>, Layout, Accessor> x)
{
  bool done = false;
  if constexpr (maybe_can_blas_scale<Scalar, decltype(x)>()) {
    done = try_blas_scale(alpha, x);
  }
  if (!done) {
    generic_scale_rank_1(alpha, x);
  }
}

template<class ElementType,
	 class IndexType,
         ::std::size_t numRows,
         ::std::size_t numCols,
         class Layout,
         class Accessor,
         class Scalar>
void linalg_scale_rank_2(
  const Scalar alpha,
  mdspan<ElementType, extents<IndexType, numRows, numCols>, Layout, Accessor> A)
{
  for (IndexType j = 0; j < A.extent(1); ++j) {
    for (IndexType i = 0; i < A.extent(0); ++i) {
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
	 class IndexType,
         ::std::size_t ... ext,
         class Layout,
         class Accessor>
void scale(
  impl::inline_exec_t&& /* exec */,
  const Scalar alpha,
  mdspan<ElementType, extents<IndexType, ext ...>, Layout, Accessor> x)
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
	 class IndexType,
         ::std::size_t ... ext,
         class Layout,
         class Accessor>
void scale(
  ExecutionPolicy&& exec,
  const Scalar alpha,
  mdspan<ElementType, extents<IndexType, ext ...>, Layout, Accessor> x)
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
	 class IndexType,
         ::std::size_t ... ext,
         class Layout,
         class Accessor>
void scale(const Scalar alpha,
           mdspan<ElementType, extents<IndexType, ext ...>, Layout, Accessor> x)
{
  scale(impl::default_exec_t{}, alpha, x);
}

} // end namespace linalg
} // end inline namespace __p1673_version_0
} // end namespace MDSPAN_IMPL_PROPOSED_NAMESPACE
} // end namespace MDSPAN_IMPL_STANDARD_NAMESPACE

#endif //LINALG_INCLUDE_EXPERIMENTAL___P1673_BITS_BLAS1_SCALE_HPP_
