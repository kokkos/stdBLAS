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

#ifndef LINALG_INCLUDE_EXPERIMENTAL___P1673_BITS_BLAS1_MATRIX_FROB_NORM_HPP_
#define LINALG_INCLUDE_EXPERIMENTAL___P1673_BITS_BLAS1_MATRIX_FROB_NORM_HPP_

#include <cmath>
#include <cstdlib>

namespace MDSPAN_IMPL_STANDARD_NAMESPACE {
namespace MDSPAN_IMPL_PROPOSED_NAMESPACE {
inline namespace __p1673_version_0 {
namespace linalg {

// begin anonymous namespace
namespace {

template <class Exec, class A_t, class Scalar, class = void>
struct is_custom_matrix_frob_norm_avail : std::false_type {};

template <class Exec, class A_t, class Scalar>
struct is_custom_matrix_frob_norm_avail<
  Exec, A_t, Scalar,
  std::enable_if_t<
    std::is_same<
      decltype(matrix_frob_norm
	       (std::declval<Exec>(),
		std::declval<A_t>(),
		std::declval<Scalar>()
		)
	       ),
      Scalar
      >::value
    && ! impl::is_inline_exec_v<Exec>
    >
  >
  : std::true_type{};
} // end anonymous namespace

template<
    class ElementType,
    class SizeType,
    ::std::size_t numRows,
    ::std::size_t numCols,
    class Layout,
    class Accessor,
    class Scalar>
Scalar matrix_frob_norm(
  impl::inline_exec_t&& /* exec */,
  mdspan<ElementType, extents<SizeType, numRows, numCols>, Layout, Accessor> A,
  Scalar init)
{
  using std::abs;
  using std::sqrt;
  using size_type = SizeType;

  // Handle special cases.
  auto result = init;
  if (A.extent(0) == 0 || A.extent(1) == 0) {
    return result;
  }
  else if(A.extent(0) == size_type(1) && A.extent(1) == size_type(1)) {
    result += abs(A(0, 0));
    return result;
  }

  // Rescaling avoids unwarranted overflow or underflow.
  Scalar scale = 0.0;
  Scalar ssq = 1.0;
  for (size_type i = 0; i < A.extent(0); ++i) {
    for (size_type j = 0; j < A.extent(1); ++j) {
      const auto absaij = abs(A(i,j));
      if (absaij != 0.0) {
        const auto quotient = scale / absaij;
        if (scale < absaij) {
          ssq = Scalar(1.0) + ssq * quotient * quotient;
          scale = absaij;
        }
        else {
          ssq = ssq + quotient * quotient;
        }
      }
    }
  }
  result += scale * sqrt(ssq);
  return result;
}

template<class ExecutionPolicy,
  class ElementType,
  class SizeType,
  ::std::size_t numRows,
  ::std::size_t numCols,
  class Layout,
  class Accessor,
  class Scalar>
Scalar matrix_frob_norm(
  ExecutionPolicy&& exec,
  mdspan<ElementType, extents<SizeType, numRows, numCols>, Layout, Accessor> A,
  Scalar init)
{
  constexpr bool use_custom = is_custom_matrix_frob_norm_avail<
    decltype(impl::map_execpolicy_with_check(exec)), decltype(A), Scalar
    >::value;

  if constexpr (use_custom) {
    return matrix_frob_norm(impl::map_execpolicy_with_check(exec), A, init);
  }
  else {
    return matrix_frob_norm(impl::inline_exec_t(), A, init);
  }
}

template<
    class ElementType,
    class SizeType,
    ::std::size_t numRows,
    ::std::size_t numCols,
    class Layout,
    class Accessor,
    class Scalar>
Scalar matrix_frob_norm(
  mdspan<
    ElementType,
    extents<SizeType, numRows, numCols>,
    Layout,
    Accessor> A,
  Scalar init)
{
  return matrix_frob_norm(impl::default_exec_t{}, A, init);
}

namespace matrix_frob_norm_detail {
  // The point of this is to do correct ADL for abs,
  // without exposing "using std::abs" in the outer namespace.
  using std::abs;
  template<
    class ElementType,
    class SizeType, ::std::size_t numRows, ::std::size_t numCols,
    class Layout,
    class Accessor>
  auto matrix_frob_norm_return_type_deducer(
    mdspan<
      ElementType,
      extents<SizeType, numRows, numCols>,
      Layout,
      Accessor
    > A) -> decltype( abs(A(0,0)) * abs(A(0,0)) );

} // namespace matrix_frob_norm_detail

template<
  class ElementType,
  class SizeType,
  ::std::size_t numRows,
  ::std::size_t numCols,
  class Layout,
  class Accessor>
auto matrix_frob_norm(
  mdspan<
    ElementType, extents<SizeType, numRows, numCols>, Layout, Accessor
  > A)
  -> decltype(matrix_frob_norm_detail::matrix_frob_norm_return_type_deducer(A))
{
  using return_t = decltype(matrix_frob_norm_detail::matrix_frob_norm_return_type_deducer(A));
  return matrix_frob_norm(A, return_t{});
}

template<
  class ExecutionPolicy,
  class ElementType,
  class SizeType,
  ::std::size_t numRows,
  ::std::size_t numCols,
  class Layout,
  class Accessor>
auto matrix_frob_norm(
  ExecutionPolicy&& exec,
  mdspan<
    ElementType,
    extents<SizeType, numRows, numCols>,
    Layout,
    Accessor
  > A)
  -> decltype(matrix_frob_norm_detail::matrix_frob_norm_return_type_deducer(A))
{
  using return_t = decltype(matrix_frob_norm_detail::matrix_frob_norm_return_type_deducer(A));
  return matrix_frob_norm(exec, A, return_t{});
}

} // end namespace linalg
} // end inline namespace __p1673_version_0
} // end namespace MDSPAN_IMPL_PROPOSED_NAMESPACE
} // end namespace MDSPAN_IMPL_STANDARD_NAMESPACE

#endif //LINALG_INCLUDE_EXPERIMENTAL___P1673_BITS_BLAS1_MATRIX_FROB_NORM_HPP_
