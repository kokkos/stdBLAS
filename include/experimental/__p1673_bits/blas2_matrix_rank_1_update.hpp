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

#ifndef LINALG_INCLUDE_EXPERIMENTAL___P1673_BITS_BLAS2_MATRIX_RANK_1_UPDATE_HPP_
#define LINALG_INCLUDE_EXPERIMENTAL___P1673_BITS_BLAS2_MATRIX_RANK_1_UPDATE_HPP_

namespace MDSPAN_IMPL_STANDARD_NAMESPACE {
namespace MDSPAN_IMPL_PROPOSED_NAMESPACE {
inline namespace __p1673_version_0 {
namespace linalg {

namespace { // (anonymous)

template <class Exec, class x_t, class y_t, class A_t, class = void>
struct is_custom_matrix_rank_1_update_avail : std::false_type {};

template <class Exec, class x_t, class y_t, class A_t>
struct is_custom_matrix_rank_1_update_avail<
  Exec, x_t, y_t, A_t,
  std::enable_if_t<
    std::is_void_v<
      decltype(matrix_rank_1_update
	       (std::declval<Exec>(),
		std::declval<x_t>(),
		std::declval<y_t>(),
		std::declval<A_t>()
		)
	       )
      >
    && ! impl::is_inline_exec_v<Exec>
    >
  >
  : std::true_type{};

#if defined(LINALG_FIX_RANK_UPDATES)

// For the overwriting case, use E_t = void.
// For the no-alpha case, use ScaleFactorType = void.
template <class Exec, class ScaleFactorType, class x_t, class E_t, class A_t, class Tr_t, class = void>
struct is_custom_symmetric_matrix_rank_1_update_avail : std::false_type
{};

// Overwriting, ExecutionPolicy != inline_exec_t, no alpha
template <class Exec, class x_t, class A_t, class Tr_t>
struct is_custom_symmetric_matrix_rank_1_update_avail<
  Exec, /* ScaleFactorType = */ void, x_t, /* E_t = */ void, A_t, Tr_t,
  std::enable_if_t<
    std::is_void_v<
      decltype(symmetric_matrix_rank_1_update(
        std::declval<Exec>(),
	std::declval<x_t>(),
        std::declval<A_t>(),
        std::declval<Tr_t>()))
    > &&
    ! impl::is_inline_exec_v<Exec>
  >
> : std::true_type
{};

// Updating, ExecutionPolicy != inline_exec_t, no alpha
template <class Exec, class x_t, class E_t, class A_t, class Tr_t>
struct is_custom_symmetric_matrix_rank_1_update_avail<
  Exec, /* ScaleFactorType = */ void, x_t, E_t, A_t, Tr_t,
  std::enable_if_t<
    std::is_void_v<
      decltype(symmetric_matrix_rank_1_update(
        std::declval<Exec>(),
	std::declval<x_t>(),
        std::declval<E_t>(), // implies not void
        std::declval<A_t>(),
        std::declval<Tr_t>()))
    > &&
    ! impl::is_inline_exec_v<Exec>
  >
> : std::true_type
{};

// Overwriting, ExecutionPolicy != inline_exec_t, alpha
template <class Exec, class ScaleFactorType, class x_t, class A_t, class Tr_t>
struct is_custom_symmetric_matrix_rank_1_update_avail<
  Exec, ScaleFactorType, x_t, /* E_t = */ void, A_t, Tr_t,
  std::enable_if_t<
    std::is_void_v<
      decltype(symmetric_matrix_rank_1_update(
        std::declval<Exec>(),
        std::declval<ScaleFactorType>(), // implies not void
        std::declval<x_t>(),
        std::declval<A_t>(),
        std::declval<Tr_t>()))
    > &&
    ! impl::is_inline_exec_v<Exec>
  >
> : std::true_type
{};

// Updating, ExecutionPolicy != inline_exec_t, alpha
template <class Exec, class ScaleFactorType, class x_t, class E_t, class A_t, class Tr_t>
struct is_custom_symmetric_matrix_rank_1_update_avail<
  Exec, ScaleFactorType, x_t, E_t, A_t, Tr_t,
  std::enable_if_t<
    std::is_void_v<
      decltype(symmetric_matrix_rank_1_update(
        std::declval<Exec>(),
        std::declval<ScaleFactorType>(), // implies not void
        std::declval<x_t>(),
        std::declval<E_t>(), // implies not void
        std::declval<A_t>(),
        std::declval<Tr_t>()))
    > &&
    ! impl::is_inline_exec_v<Exec>
  >
> : std::true_type
{};

#else

template <class Exec, class ScaleFactorType, class x_t, class A_t, class Tr_t, class = void>
struct is_custom_symmetric_matrix_rank_1_update_avail : std::false_type
{};

// ExecutionPolicy != inline_exec_t, no alpha
template <class Exec, class x_t, class A_t, class Tr_t>
struct is_custom_symmetric_matrix_rank_1_update_avail<
  Exec, void, x_t, A_t, Tr_t,
  std::enable_if_t<
    std::is_void_v<
      decltype(symmetric_matrix_rank_1_update
	       (std::declval<Exec>(),
		std::declval<x_t>(),
		std::declval<A_t>(),
		std::declval<Tr_t>()
		)
	       )
      >
    && ! impl::is_inline_exec_v<Exec>
    >
  >
  : std::true_type
{};

// ExecutionPolicy != inline_exec_t,  alpha
template <class Exec, class ScaleFactorType, class x_t, class A_t, class Tr_t>
struct is_custom_symmetric_matrix_rank_1_update_avail<
  Exec, ScaleFactorType, x_t, A_t, Tr_t,
  std::enable_if_t<
    std::is_void_v<
      decltype(symmetric_matrix_rank_1_update
	       (std::declval<Exec>(),
                std::declval<ScaleFactorType>(),
		std::declval<x_t>(),
		std::declval<A_t>(),
		std::declval<Tr_t>()
		)
	       )
      >
    && ! impl::is_inline_exec_v<Exec>
    >
  >
  : std::true_type
{};

#endif // LINALG_FIX_RANK_UPDATES

#if defined(LINALG_FIX_RANK_UPDATES)

// For the overwriting case, use E_t = void.
// For the no-alpha case, use ScaleFactorType = void.
template <class Exec, class ScaleFactorType, class x_t, class E_t, class A_t, class Tr_t, class = void>
struct is_custom_hermitian_matrix_rank_1_update_avail : std::false_type
{};

// Overwriting, ExecutionPolicy != inline_exec_t, no alpha
template <class Exec, class x_t, class A_t, class Tr_t>
struct is_custom_hermitian_matrix_rank_1_update_avail<
  Exec, /* ScaleFactorType = */ void, x_t, /* E_t = */ void, A_t, Tr_t,
  std::enable_if_t<
    std::is_void_v<
      decltype(hermitian_matrix_rank_1_update(
        std::declval<Exec>(),
	std::declval<x_t>(),
        std::declval<A_t>(),
        std::declval<Tr_t>()))
    > &&
    ! impl::is_inline_exec_v<Exec>
  >
> : std::true_type
{};

// Updating, ExecutionPolicy != inline_exec_t, no alpha
template <class Exec, class x_t, class E_t, class A_t, class Tr_t>
struct is_custom_hermitian_matrix_rank_1_update_avail<
  Exec, /* ScaleFactorType = */ void, x_t, E_t, A_t, Tr_t,
  std::enable_if_t<
    std::is_void_v<
      decltype(hermitian_matrix_rank_1_update(
        std::declval<Exec>(),
	std::declval<x_t>(),
        std::declval<E_t>(), // implies not void
        std::declval<A_t>(),
        std::declval<Tr_t>()))
    > &&
    ! impl::is_inline_exec_v<Exec>
  >
> : std::true_type
{};

// Overwriting, ExecutionPolicy != inline_exec_t, alpha
template <class Exec, class ScaleFactorType, class x_t, class A_t, class Tr_t>
struct is_custom_hermitian_matrix_rank_1_update_avail<
  Exec, ScaleFactorType, x_t, /* E_t = */ void, A_t, Tr_t,
  std::enable_if_t<
    std::is_void_v<
      decltype(hermitian_matrix_rank_1_update(
        std::declval<Exec>(),
        std::declval<ScaleFactorType>(), // implies not void
        std::declval<x_t>(),
        std::declval<A_t>(),
        std::declval<Tr_t>()))
    > &&
    ! impl::is_inline_exec_v<Exec>
  >
> : std::true_type
{};

// Updating, ExecutionPolicy != inline_exec_t, alpha
template <class Exec, class ScaleFactorType, class x_t, class E_t, class A_t, class Tr_t>
struct is_custom_hermitian_matrix_rank_1_update_avail<
  Exec, ScaleFactorType, x_t, E_t, A_t, Tr_t,
  std::enable_if_t<
    std::is_void_v<
      decltype(hermitian_matrix_rank_1_update(
        std::declval<Exec>(),
        std::declval<ScaleFactorType>(), // implies not void
        std::declval<x_t>(),
        std::declval<E_t>(), // implies not void
        std::declval<A_t>(),
        std::declval<Tr_t>()))
    > &&
    ! impl::is_inline_exec_v<Exec>
  >
> : std::true_type
{};

#else

template <class Exec, class ScaleFactorType, class x_t, class A_t, class Tr_t, class = void>
struct is_custom_hermitian_matrix_rank_1_update_avail : std::false_type
{};

// ExecutionPolicy != inline_exec_t, no alpha
template <class Exec, class x_t, class A_t, class Tr_t>
struct is_custom_hermitian_matrix_rank_1_update_avail<
  Exec, /* ScaleFactorType = */ void, x_t, A_t, Tr_t,
  std::enable_if_t<
    std::is_void_v<
      decltype(hermitian_matrix_rank_1_update(
        std::declval<Exec>(),
	std::declval<x_t>(),
        std::declval<A_t>(),
        std::declval<Tr_t>()))
    > &&
    ! impl::is_inline_exec_v<Exec>
  >
> : std::true_type
{};

// ExecutionPolicy != inline_exec_t, alpha
template <class Exec, class ScaleFactorType, class x_t, class A_t, class Tr_t>
struct is_custom_hermitian_matrix_rank_1_update_avail<
  Exec, ScaleFactorType, x_t, A_t, Tr_t,
  std::enable_if_t<
    std::is_void_v<
      decltype(hermitian_matrix_rank_1_update(
        std::declval<Exec>(),
        std::declval<ScaleFactorType>(),
        std::declval<x_t>(),
        std::declval<A_t>(),
        std::declval<Tr_t>()))
    >
    && ! impl::is_inline_exec_v<Exec>
  >
> : std::true_type
{};

#endif

} // namespace (anonymous)

// Nonsymmetric non-conjugated rank-1 update

template<class ElementType_x,
         class SizeType_x, ::std::size_t ext_x,
         class Layout_x,
         class Accessor_x,
         class ElementType_y,
         class SizeType_y, ::std::size_t ext_y,
         class Layout_y,
         class Accessor_y,
         class ElementType_A,
         class SizeType_A, ::std::size_t numRows_A,
         ::std::size_t numCols_A,
         class Layout_A,
         class Accessor_A>
void matrix_rank_1_update(
  impl::inline_exec_t&& /* exec */,
  mdspan<ElementType_x, extents<SizeType_x, ext_x>, Layout_x, Accessor_x> x,
  mdspan<ElementType_y, extents<SizeType_y, ext_y>, Layout_y, Accessor_y> y,
  mdspan<ElementType_A, extents<SizeType_A, numRows_A, numCols_A>, Layout_A, Accessor_A> A)
{
  using size_type = ::std::common_type_t<SizeType_x, SizeType_y, SizeType_A>;

  for (size_type i = 0; i < A.extent(0); ++i) {
    for (size_type j = 0; j < A.extent(1); ++j) {
      A(i,j) += x(i) * y(j);
    }
  }
}

template<class ExecutionPolicy,
         class ElementType_x,
         class SizeType_x, ::std::size_t ext_x,
         class Layout_x,
         class Accessor_x,
         class ElementType_y,
         class SizeType_y, ::std::size_t ext_y,
         class Layout_y,
         class Accessor_y,
         class ElementType_A,
         class SizeType_A, ::std::size_t numRows_A,
         ::std::size_t numCols_A,
         class Layout_A,
         class Accessor_A>
void matrix_rank_1_update(
  ExecutionPolicy&& exec,
  mdspan<ElementType_x, extents<SizeType_x, ext_x>, Layout_x, Accessor_x> x,
  mdspan<ElementType_y, extents<SizeType_y, ext_y>, Layout_y, Accessor_y> y,
  mdspan<ElementType_A, extents<SizeType_A, numRows_A, numCols_A>, Layout_A, Accessor_A> A)
{
  constexpr bool use_custom = is_custom_matrix_rank_1_update_avail<
    decltype(impl::map_execpolicy_with_check(exec)), decltype(x), decltype(y), decltype(A)
    >::value;

  if constexpr (use_custom) {
    matrix_rank_1_update(impl::map_execpolicy_with_check(exec), x, y, A);
  }
  else {
    matrix_rank_1_update(impl::inline_exec_t{}, x, y, A);
  }
}

template<class ElementType_x,
         class SizeType_x, ::std::size_t ext_x,
         class Layout_x,
         class Accessor_x,
         class ElementType_y,
         class SizeType_y, ::std::size_t ext_y,
         class Layout_y,
         class Accessor_y,
         class ElementType_A,
         class SizeType_A, ::std::size_t numRows_A,
         ::std::size_t numCols_A,
         class Layout_A,
         class Accessor_A>
void matrix_rank_1_update(
  mdspan<ElementType_x, extents<SizeType_x, ext_x>, Layout_x, Accessor_x> x,
  mdspan<ElementType_y, extents<SizeType_y, ext_y>, Layout_y, Accessor_y> y,
  mdspan<ElementType_A, extents<SizeType_A, numRows_A, numCols_A>, Layout_A, Accessor_A> A)
{
  matrix_rank_1_update(impl::default_exec_t{}, x, y, A);
}


// Nonsymmetric conjugated rank-1 update

template<class ElementType_x,
         class SizeType_x, ::std::size_t ext_x,
         class Layout_x,
         class Accessor_x,
         class ElementType_y,
         class SizeType_y, ::std::size_t ext_y,
         class Layout_y,
         class Accessor_y,
         class ElementType_A,
         class SizeType_A, ::std::size_t numRows_A,
         ::std::size_t numCols_A,
         class Layout_A,
         class Accessor_A>
void matrix_rank_1_update_c(
  mdspan<ElementType_x, extents<SizeType_x, ext_x>, Layout_x, Accessor_x> x,
  mdspan<ElementType_y, extents<SizeType_y, ext_y>, Layout_y, Accessor_y> y,
  mdspan<ElementType_A, extents<SizeType_A, numRows_A, numCols_A>, Layout_A, Accessor_A> A)
{
  matrix_rank_1_update(x, conjugated(y), A);
}

template<class ExecutionPolicy,
         class ElementType_x,
         class SizeType_x, ::std::size_t ext_x,
         class Layout_x,
         class Accessor_x,
         class ElementType_y,
         class SizeType_y, ::std::size_t ext_y,
         class Layout_y,
         class Accessor_y,
         class ElementType_A,
         class SizeType_A, ::std::size_t numRows_A,
         ::std::size_t numCols_A,
         class Layout_A,
         class Accessor_A>
void matrix_rank_1_update_c(
  ExecutionPolicy&& exec,
  mdspan<ElementType_x, extents<SizeType_x, ext_x>, Layout_x, Accessor_x> x,
  mdspan<ElementType_y, extents<SizeType_y, ext_y>, Layout_y, Accessor_y> y,
  mdspan<ElementType_A, extents<SizeType_A, numRows_A, numCols_A>, Layout_A, Accessor_A> A)
{
  matrix_rank_1_update(exec, x, conjugated(y), A);
}

// Symmetric matrix rank-1 update

// Overwriting symmetric rank-1 matrix update
// (inline_exec_t, alpha)
MDSPAN_TEMPLATE_REQUIRES(
  class ScaleFactorType,
  class ElementType_x,
  class SizeType_x, ::std::size_t ext_x,
  class Layout_x,
  class Accessor_x,
  class ElementType_A,
  class SizeType_A, ::std::size_t numRows_A,
  ::std::size_t numCols_A,
  class Layout_A,
  class Accessor_A,
  class Triangle,
  /* requires */ (
    std::is_same_v<Triangle, lower_triangle_t> ||
    std::is_same_v<Triangle, upper_triangle_t>
  )
)
void symmetric_matrix_rank_1_update(
  impl::inline_exec_t&& /* exec */,
  ScaleFactorType alpha,
  mdspan<ElementType_x, extents<SizeType_x, ext_x>, Layout_x, Accessor_x> x,
  mdspan<ElementType_A, extents<SizeType_A, numRows_A, numCols_A>, Layout_A, Accessor_A> A,
  Triangle /* t */)
{
  using index_type = std::common_type_t<SizeType_x, SizeType_A>;

  if constexpr (std::is_same_v<Triangle, lower_triangle_t>) {
    for (index_type i = 0; i < A.extent(0); ++i) {
      for (index_type j = 0; j <= i; ++j) {
#if defined(LINALG_FIX_RANK_UPDATES)
        A(i,j) = typename decltype(A)::value_type{};
#endif // LINALG_FIX_RANK_UPDATES
        A(i,j) += alpha * x(i) * x(j);
      }
    }
  }
  else { // upper triangle
    for (index_type j = 0; j < A.extent(1); ++j) {
      for (index_type i = 0; i <= j; ++i) {
#if defined(LINALG_FIX_RANK_UPDATES)
        A(i,j) = typename decltype(A)::value_type{};
#endif // LINALG_FIX_RANK_UPDATES
        A(i,j) += alpha * x(i) * x(j);
      }
    }
  }
}

#if defined(LINALG_FIX_RANK_UPDATES)
// Updating symmetric rank-1 matrix update
// (inline_exec_t, alpha)
MDSPAN_TEMPLATE_REQUIRES(
  class ScaleFactorType,
  class ElementType_x,
  class SizeType_x, ::std::size_t ext_x,
  class Layout_x,
  class Accessor_x,
  class ElementType_E,
  class SizeType_E, ::std::size_t numRows_E,
  ::std::size_t numCols_E,
  class Layout_E,
  class Accessor_E,
  class ElementType_A,
  class SizeType_A, ::std::size_t numRows_A,
  ::std::size_t numCols_A,
  class Layout_A, // TODO Implement Constraints and Mandates
  class Accessor_A,
  class Triangle,
  /* requires */ (
    std::is_same_v<Triangle, lower_triangle_t> ||
    std::is_same_v<Triangle, upper_triangle_t>
  )
)
void symmetric_matrix_rank_1_update(
  impl::inline_exec_t&& /* exec */,
  ScaleFactorType alpha,
  mdspan<ElementType_x, extents<SizeType_x, ext_x>, Layout_x, Accessor_x> x,
  mdspan<ElementType_E, extents<SizeType_E, numRows_E, numCols_E>, Layout_E, Accessor_E> E,
  mdspan<ElementType_A, extents<SizeType_A, numRows_A, numCols_A>, Layout_A, Accessor_A> A,
  Triangle /* t */)
{
  using index_type = std::common_type_t<SizeType_x, SizeType_E, SizeType_A>;

  if constexpr (std::is_same_v<Triangle, lower_triangle_t>) {
    for (index_type j = 0; j < A.extent(1); ++j) {
      for (index_type i = j; i < A.extent(0); ++i) {
        A(i,j) = E(i,j) + alpha * x(i) * x(j);
      }
    }
  }
  else {
    for (index_type j = 0; j < A.extent(1); ++j) {
      for (index_type i = 0; i <= j; ++i) {
        A(i,j) = E(i,j) + alpha * x(i) * x(j);
      }
    }
  }
}
#endif // LINALG_FIX_RANK_UPDATES

// Overwriting symmetric rank-1 matrix update
// (ExecutionPolicy&&, alpha)
MDSPAN_TEMPLATE_REQUIRES(
  class ExecutionPolicy,
  class ScaleFactorType,
  class ElementType_x,
  class SizeType_x, ::std::size_t ext_x,
  class Layout_x,
  class Accessor_x,
  class ElementType_A,
  class SizeType_A, ::std::size_t numRows_A,
  ::std::size_t numCols_A,
  class Layout_A,
  class Accessor_A,
  class Triangle,
  /* requires */ (
    impl::is_linalg_execution_policy_other_than_inline_v<ExecutionPolicy> &&
    (std::is_same_v<Triangle, lower_triangle_t> ||
     std::is_same_v<Triangle, upper_triangle_t>)
  )
)
void symmetric_matrix_rank_1_update(
  ExecutionPolicy&& exec,
  ScaleFactorType alpha,
  mdspan<ElementType_x, extents<SizeType_x, ext_x>, Layout_x, Accessor_x> x,
  mdspan<ElementType_A, extents<SizeType_A, numRows_A, numCols_A>, Layout_A, Accessor_A> A,
  Triangle t)
{
  constexpr bool use_custom = is_custom_symmetric_matrix_rank_1_update_avail<
      decltype(execpolicy_mapper(exec)),
      ScaleFactorType,
      decltype(x),
#if defined(LINALG_FIX_RANK_UPDATES)
      /* decltype(E) = */ void,
#endif // LINALG_FIX_RANK_UPDATES
      decltype(A),
      Triangle
    >::value;

  if constexpr (use_custom) {
    symmetric_matrix_rank_1_update(execpolicy_mapper(exec), alpha, x, A, t);
  }
  else {
    symmetric_matrix_rank_1_update(impl::inline_exec_t{}, alpha, x, A, t);
  }
}

#if defined(LINALG_FIX_RANK_UPDATES)
// Updating symmetric rank-1 matrix update
// (ExecutionPolicy&&, alpha)
MDSPAN_TEMPLATE_REQUIRES(
  class ExecutionPolicy,
  class ScaleFactorType,
  class ElementType_x,
  class SizeType_x, ::std::size_t ext_x,
  class Layout_x,
  class Accessor_x,
  class ElementType_E,
  // Work-around for MDSPAN_TEMPLATE_REQUIRES not taking too many arguments:
  // combine (SizeType_E, numRows_E, numCols_E) into Extents_E
  // and add rank() == 2 constraint.
  class Extents_E,
  class Layout_E,
  class Accessor_E,
  class ElementType_A,
  class SizeType_A, ::std::size_t numRows_A,
  ::std::size_t numCols_A,
  class Layout_A, // TODO Implement Constraints and Mandates
  class Accessor_A,
  class Triangle,
  /* requires */ (
    impl::is_linalg_execution_policy_other_than_inline_v<ExecutionPolicy> &&
    (std::is_same_v<Triangle, lower_triangle_t> ||
     std::is_same_v<Triangle, upper_triangle_t>) &&
    Extents_E::rank() == 2
  )
)
void symmetric_matrix_rank_1_update(
  ExecutionPolicy&& exec,
  ScaleFactorType alpha,
  mdspan<ElementType_x, extents<SizeType_x, ext_x>, Layout_x, Accessor_x> x,
  mdspan<ElementType_E, Extents_E,                                 Layout_E, Accessor_E> E,
  mdspan<ElementType_A, extents<SizeType_A, numRows_A, numCols_A>, Layout_A, Accessor_A> A,
  Triangle t)
{
  constexpr bool use_custom = is_custom_symmetric_matrix_rank_1_update_avail<
      decltype(execpolicy_mapper(exec)),
      ScaleFactorType,
      decltype(x),
      decltype(E),
      decltype(A),
      Triangle
    >::value;

  if constexpr (use_custom) {
    symmetric_matrix_rank_1_update(execpolicy_mapper(exec), alpha, x, E, A, t);
  }
  else {
    symmetric_matrix_rank_1_update(impl::inline_exec_t{}, alpha, x, E, A, t);
  }
}
#endif // LINALG_FIX_RANK_UPDATES

// Overwriting symmetric rank-1 matrix update
// (no execution policy, alpha)
MDSPAN_TEMPLATE_REQUIRES(
  class ScaleFactorType,
  class ElementType_x,
  class SizeType_x, ::std::size_t ext_x,
  class Layout_x,
  class Accessor_x,
  class ElementType_A,
  class SizeType_A, ::std::size_t numRows_A,
  ::std::size_t numCols_A,
  class Layout_A,
  class Accessor_A,
  class Triangle,
  /* requires */ (
    (! impl::is_linalg_execution_policy_other_than_inline_v<ScaleFactorType>) &&
    (std::is_same_v<Triangle, lower_triangle_t> ||
     std::is_same_v<Triangle, upper_triangle_t>)
  )
)
void symmetric_matrix_rank_1_update(
  ScaleFactorType alpha,
  mdspan<ElementType_x, extents<SizeType_x, ext_x>, Layout_x, Accessor_x> x,
  mdspan<ElementType_A, extents<SizeType_A, numRows_A, numCols_A>, Layout_A, Accessor_A> A,
  Triangle t)
{
  symmetric_matrix_rank_1_update(impl::default_exec_t{}, alpha, x, A, t);
}

#if defined(LINALG_FIX_RANK_UPDATES)
// Updating symmetric rank-1 matrix update
// (no execution policy, alpha)
MDSPAN_TEMPLATE_REQUIRES(
  class ScaleFactorType,
  class ElementType_x,
  class SizeType_x, ::std::size_t ext_x,
  class Layout_x,
  class Accessor_x,
  class ElementType_E,
  class SizeType_E, ::std::size_t numRows_E,
  ::std::size_t numCols_E,
  class Layout_E,
  class Accessor_E,
  class ElementType_A,
  class SizeType_A, ::std::size_t numRows_A,
  ::std::size_t numCols_A,
  class Layout_A, // TODO Implement Constraints and Mandates
  class Accessor_A,
  class Triangle,
  /* requires */ (
    (! impl::is_linalg_execution_policy_other_than_inline_v<ScaleFactorType>) &&
    (std::is_same_v<Triangle, lower_triangle_t> ||
     std::is_same_v<Triangle, upper_triangle_t>)
  )
)
void symmetric_matrix_rank_1_update(
  ScaleFactorType alpha,
  mdspan<ElementType_x, extents<SizeType_x, ext_x>, Layout_x, Accessor_x> x,
  mdspan<ElementType_E, extents<SizeType_E, numRows_E, numCols_E>, Layout_E, Accessor_E> E,
  mdspan<ElementType_A, extents<SizeType_A, numRows_A, numCols_A>, Layout_A, Accessor_A> A,
  Triangle t)
{
  symmetric_matrix_rank_1_update(impl::default_exec_t{}, alpha, x, E, A, t);
}
#endif // LINALG_FIX_RANK_UPDATES

// Overwriting symmetric rank-1 matrix update
// (inline_exec_t, no alpha)
MDSPAN_TEMPLATE_REQUIRES(
  class ElementType_x,
  class SizeType_x, ::std::size_t ext_x,
  class Layout_x,
  class Accessor_x,
  class ElementType_A,
  class SizeType_A, ::std::size_t numRows_A,
  ::std::size_t numCols_A,
  class Layout_A,
  class Accessor_A,
  class Triangle,
  /* requires */ (
    std::is_same_v<Triangle, lower_triangle_t> ||
    std::is_same_v<Triangle, upper_triangle_t>
  )
)
void symmetric_matrix_rank_1_update(
  impl::inline_exec_t&& /* exec */,
  mdspan<ElementType_x, extents<SizeType_x, ext_x>, Layout_x, Accessor_x> x,
  mdspan<ElementType_A, extents<SizeType_A, numRows_A, numCols_A>, Layout_A, Accessor_A> A,
  Triangle /* t */)
{
  using index_type = std::common_type_t<SizeType_x, SizeType_A>;

  if constexpr (std::is_same_v<Triangle, lower_triangle_t>) {
    for (index_type i = 0; i < A.extent(0); ++i) {
      for (index_type j = 0; j <= i; ++j) {
#if defined(LINALG_FIX_RANK_UPDATES)
        A(i,j) = x(i) * x(j);
#else
        A(i,j) += x(i) * x(j);
#endif // LINALG_FIX_RANK_UPDATES
      }
    }
  }
  else { // upper triangle
    for (index_type j = 0; j < A.extent(1); ++j) {
      for (index_type i = 0; i <= j; ++i) {
#if defined(LINALG_FIX_RANK_UPDATES)
        A(i,j) = typename decltype(A)::value_type{};
#endif // LINALG_FIX_RANK_UPDATES
        A(i,j) += x(i) * x(j);
      }
    }
  }
}

#if defined(LINALG_FIX_RANK_UPDATES)
// Updating symmetric rank-1 matrix update
// (inline_exec_t, no alpha)
MDSPAN_TEMPLATE_REQUIRES(
  class ElementType_x,
  class SizeType_x, ::std::size_t ext_x,
  class Layout_x,
  class Accessor_x,
  class ElementType_E,
  class SizeType_E, ::std::size_t numRows_E,
  ::std::size_t numCols_E,
  class Layout_E,
  class Accessor_E,
  class ElementType_A,
  class SizeType_A, ::std::size_t numRows_A,
  ::std::size_t numCols_A,
  class Layout_A,
  class Accessor_A,
  class Triangle,
  /* requires */ (
    std::is_same_v<Triangle, lower_triangle_t> ||
    std::is_same_v<Triangle, upper_triangle_t>
  )
)
void symmetric_matrix_rank_1_update(
  impl::inline_exec_t&& /* exec */,
  mdspan<ElementType_x, extents<SizeType_x, ext_x>, Layout_x, Accessor_x> x,
  mdspan<ElementType_E, extents<SizeType_E, numRows_E, numCols_E>, Layout_E, Accessor_E> E,
  mdspan<ElementType_A, extents<SizeType_A, numRows_A, numCols_A>, Layout_A, Accessor_A> A,
  Triangle /* t */)
{
  using index_type = std::common_type_t<SizeType_x, SizeType_E, SizeType_A>;

  if constexpr (std::is_same_v<Triangle, lower_triangle_t>) {
    for (index_type i = 0; i < A.extent(0); ++i) {
      for (index_type j = 0; j <= i; ++j) {
        A(i,j) = E(i,j) + x(i) * x(j);
      }
    }
  }
  else { // upper triangle
    for (index_type j = 0; j < A.extent(1); ++j) {
      for (index_type i = 0; i <= j; ++i) {
        A(i,j) = E(i,j) + x(i) * x(j);
      }
    }
  }
}
#endif // LINALG_FIX_RANK_UPDATES

// Overwriting symmetric rank-1 matrix update
// (ExecutionPolicy&&, no alpha)
MDSPAN_TEMPLATE_REQUIRES(
  class ExecutionPolicy,
  class ElementType_x,
  class SizeType_x, ::std::size_t ext_x,
  class Layout_x,
  class Accessor_x,
  class ElementType_A,
  class SizeType_A, ::std::size_t numRows_A,
  ::std::size_t numCols_A,
  class Layout_A,
  class Accessor_A,
  class Triangle,
  /* requires */ (
    impl::is_linalg_execution_policy_other_than_inline_v<ExecutionPolicy> &&
    (std::is_same_v<Triangle, lower_triangle_t> ||
     std::is_same_v<Triangle, upper_triangle_t>)
  )
)
void symmetric_matrix_rank_1_update(
  ExecutionPolicy&& exec,
  mdspan<ElementType_x, extents<SizeType_x, ext_x>, Layout_x, Accessor_x> x,
  mdspan<ElementType_A, extents<SizeType_A, numRows_A, numCols_A>, Layout_A, Accessor_A> A,
  Triangle t)
{
  constexpr bool use_custom = is_custom_symmetric_matrix_rank_1_update_avail<
      decltype(impl::map_execpolicy_with_check(exec)),
      /* ScaleFactorType = */ void,
      decltype(x),
#if defined(LINALG_FIX_RANK_UPDATES)
      /* decltype(E) = */ void,
#endif // LINALG_FIX_RANK_UPDATES
      decltype(A),
      Triangle
    >::value;

  if constexpr (use_custom) {
    symmetric_matrix_rank_1_update(impl::map_execpolicy_with_check(exec), x, A, t);
  }
  else {
    symmetric_matrix_rank_1_update(impl::inline_exec_t{}, x, A, t);
  }
}

#if defined(LINALG_FIX_RANK_UPDATES)
// Updating symmetric rank-1 matrix update
// (ExecutionPolicy&&, no alpha)
MDSPAN_TEMPLATE_REQUIRES(
  class ExecutionPolicy,
  class ElementType_x,
  class SizeType_x, ::std::size_t ext_x,
  class Layout_x,
  class Accessor_x,
  class ElementType_E,
  class SizeType_E, ::std::size_t numRows_E,
  ::std::size_t numCols_E,
  class Layout_E,
  class Accessor_E,
  class ElementType_A,
  class SizeType_A, ::std::size_t numRows_A,
  ::std::size_t numCols_A,
  class Layout_A,
  class Accessor_A,
  class Triangle,
  /* requires */ (
    impl::is_linalg_execution_policy_other_than_inline_v<ExecutionPolicy> &&
    (std::is_same_v<Triangle, lower_triangle_t> ||
     std::is_same_v<Triangle, upper_triangle_t>)
  )
)
void symmetric_matrix_rank_1_update(
  ExecutionPolicy&& exec,
  mdspan<ElementType_x, extents<SizeType_x, ext_x>, Layout_x, Accessor_x> x,
  mdspan<ElementType_E, extents<SizeType_E, numRows_E, numCols_E>, Layout_E, Accessor_E> E,
  mdspan<ElementType_A, extents<SizeType_A, numRows_A, numCols_A>, Layout_A, Accessor_A> A,
  Triangle t)
{
  constexpr bool use_custom = is_custom_symmetric_matrix_rank_1_update_avail<
      decltype(impl::map_execpolicy_with_check(exec)),
      /* ScaleFactorType = */ void,
      decltype(x),
      decltype(E),
      decltype(A),
      Triangle
    >::value;

  if constexpr (use_custom) {
    symmetric_matrix_rank_1_update(impl::map_execpolicy_with_check(exec), x, E, A, t);
  }
  else {
    symmetric_matrix_rank_1_update(impl::inline_exec_t{}, x, E, A, t);
  }
}
#endif // LINALG_FIX_RANK_UPDATES

// Overwriting symmetric rank-1 matrix update
// (no execution policy, no alpha)
MDSPAN_TEMPLATE_REQUIRES(
  class ElementType_x,
  class SizeType_x, ::std::size_t ext_x,
  class Layout_x,
  class Accessor_x,
  class ElementType_A,
  class SizeType_A, ::std::size_t numRows_A,
  ::std::size_t numCols_A,
  class Layout_A,
  class Accessor_A,
  class Triangle,
  /* requires */ (
    std::is_same_v<Triangle, lower_triangle_t> ||
    std::is_same_v<Triangle, upper_triangle_t>
  )
)
void symmetric_matrix_rank_1_update(
  mdspan<ElementType_x, extents<SizeType_x, ext_x>, Layout_x, Accessor_x> x,
  mdspan<ElementType_A, extents<SizeType_A, numRows_A, numCols_A>, Layout_A, Accessor_A> A,
  Triangle t)
{
  symmetric_matrix_rank_1_update(impl::default_exec_t{}, x, A, t);
}

#if defined(LINALG_FIX_RANK_UPDATES)
// Updating symmetric rank-1 matrix update
// (no execution policy, no alpha)
MDSPAN_TEMPLATE_REQUIRES(
  class ElementType_x,
  class SizeType_x, ::std::size_t ext_x,
  class Layout_x,
  class Accessor_x,
  class ElementType_E,
  class SizeType_E, ::std::size_t numRows_E,
  ::std::size_t numCols_E,
  class Layout_E,
  class Accessor_E,
  class ElementType_A,
  class SizeType_A, ::std::size_t numRows_A,
  ::std::size_t numCols_A,
  class Layout_A,
  class Accessor_A,
  class Triangle,
  /* requires */ (
    std::is_same_v<Triangle, lower_triangle_t> ||
    std::is_same_v<Triangle, upper_triangle_t>
  )
)
void symmetric_matrix_rank_1_update(
  mdspan<ElementType_x, extents<SizeType_x, ext_x>, Layout_x, Accessor_x> x,
  mdspan<ElementType_E, extents<SizeType_E, numRows_E, numCols_E>, Layout_E, Accessor_E> E,
  mdspan<ElementType_A, extents<SizeType_A, numRows_A, numCols_A>, Layout_A, Accessor_A> A,
  Triangle t)
{
  symmetric_matrix_rank_1_update(impl::default_exec_t{}, x, E, A, t);
}
#endif // LINALG_FIX_RANK_UPDATES

// Hermitian matrix rank-1 update

// Overwriting Hermitian rank-1 matrix update
// (inline_exec_t, alpha)
MDSPAN_TEMPLATE_REQUIRES(
  class ScaleFactorType,
  class ElementType_x,
  class SizeType_x, ::std::size_t ext_x,
  class Layout_x,
  class Accessor_x,
  class ElementType_A,
  class SizeType_A, ::std::size_t numRows_A,
  ::std::size_t numCols_A,
  class Layout_A, // TODO Implement Constraints and Mandates
  class Accessor_A,
  class Triangle,
  /* requires */ (
    std::is_same_v<Triangle, lower_triangle_t> ||
    std::is_same_v<Triangle, upper_triangle_t>
  )
)
void hermitian_matrix_rank_1_update(
  impl::inline_exec_t&& /* exec */,
  ScaleFactorType alpha,
  mdspan<ElementType_x, extents<SizeType_x, ext_x>, Layout_x, Accessor_x> x,
  mdspan<ElementType_A, extents<SizeType_A, numRows_A, numCols_A>, Layout_A, Accessor_A> A,
  Triangle /* t */)
{
  using index_type = std::common_type_t<SizeType_x, SizeType_A>;

  if constexpr (std::is_same_v<Triangle, lower_triangle_t>) {
    for (index_type i = 0; i < A.extent(0); ++i) {
      for (index_type j = 0; j <= i; ++j) {
#if defined(LINALG_FIX_RANK_UPDATES)
        A(i,j) = alpha * x(i) * impl::conj_if_needed(x(j));
#else
        auto A_ij = i == j ? impl::real_if_needed(A(i,j)) : A(i,j);
        A(i,j) = A_ij + alpha * x(i) * impl::conj_if_needed(x(j));
#endif // LINALG_FIX_RANK_UPDATES
      }
    }
  }
  else { // upper triangle
    for (index_type j = 0; j < A.extent(1); ++j) {
      for (index_type i = 0; i <= j; ++i) {
#if defined(LINALG_FIX_RANK_UPDATES)
        A(i,j) = typename decltype(A)::value_type{};
#else
        if (i == j) {
          A(j,j) = impl::real_if_needed(A(j,j));
        }
#endif // LINALG_FIX_RANK_UPDATES
        A(i,j) += alpha * x(i) * impl::conj_if_needed(x(j));
      }
    }
  }
}

#if defined(LINALG_FIX_RANK_UPDATES)
// Updating Hermitian rank-1 matrix update
// (inline_exec_t, alpha)
MDSPAN_TEMPLATE_REQUIRES(
  class ScaleFactorType,
  class ElementType_x,
  class SizeType_x, ::std::size_t ext_x,
  class Layout_x,
  class Accessor_x,
  class ElementType_E,
  class SizeType_E, ::std::size_t numRows_E,
  ::std::size_t numCols_E,
  class Layout_E,
  class Accessor_E,
  class ElementType_A,
  class SizeType_A, ::std::size_t numRows_A,
  ::std::size_t numCols_A,
  class Layout_A, // TODO Implement Constraints and Mandates
  class Accessor_A,
  class Triangle,
  /* requires */ (
    std::is_same_v<Triangle, lower_triangle_t> ||
    std::is_same_v<Triangle, upper_triangle_t>
  )
)
void hermitian_matrix_rank_1_update(
  impl::inline_exec_t&& /* exec */,
  ScaleFactorType alpha,
  mdspan<ElementType_x, extents<SizeType_x, ext_x>, Layout_x, Accessor_x> x,
  mdspan<ElementType_E, extents<SizeType_E, numRows_E, numCols_E>, Layout_E, Accessor_E> E,
  mdspan<ElementType_A, extents<SizeType_A, numRows_A, numCols_A>, Layout_A, Accessor_A> A,
  Triangle /* t */)
{
  using index_type = std::common_type_t<SizeType_x, SizeType_E, SizeType_A>;

  if constexpr (std::is_same_v<Triangle, lower_triangle_t>) {
    for (index_type j = 0; j < A.extent(1); ++j) {
      for (index_type i = j; i < A.extent(0); ++i) {
        if (i == j) {
          A(j,j) = impl::real_if_needed(E(j,j));
        }
        else {
          A(i,j) = E(i,j);
        }
        A(i,j) += alpha * x(i) * impl::conj_if_needed(x(j));
      }
    }
  }
  else {
    for (index_type j = 0; j < A.extent(1); ++j) {
      for (index_type i = 0; i <= j; ++i) {
        if (i == j) {
          A(j,j) = impl::real_if_needed(E(j,j));
        }
        else {
          A(i,j) = E(i,j);
        }
        A(i,j) += alpha * x(i) * impl::conj_if_needed(x(j));
      }
    }
  }
}
#endif // LINALG_FIX_RANK_UPDATES

// Overwriting Hermitian rank-1 matrix update
// (ExecutionPolicy&&, alpha)
MDSPAN_TEMPLATE_REQUIRES(
  class ExecutionPolicy,
  class ScaleFactorType,
  class ElementType_x,
  class SizeType_x, ::std::size_t ext_x,
  class Layout_x,
  class Accessor_x,
  class ElementType_A,
  class SizeType_A, ::std::size_t numRows_A,
  ::std::size_t numCols_A,
  class Layout_A, // TODO Implement Constraints and Mandates
  class Accessor_A,
  class Triangle,
  /* requires */ (
    impl::is_linalg_execution_policy_other_than_inline_v<ExecutionPolicy> &&
    (std::is_same_v<Triangle, lower_triangle_t> ||
     std::is_same_v<Triangle, upper_triangle_t>)
  )
)
void hermitian_matrix_rank_1_update(
  ExecutionPolicy&& exec,
  ScaleFactorType alpha,
  mdspan<ElementType_x, extents<SizeType_x, ext_x>, Layout_x, Accessor_x> x,
  mdspan<ElementType_A, extents<SizeType_A, numRows_A, numCols_A>, Layout_A, Accessor_A> A,
  Triangle t)
{
  constexpr bool use_custom = is_custom_hermitian_matrix_rank_1_update_avail<
      decltype(execpolicy_mapper(exec)),
      ScaleFactorType,
      decltype(x),
#if defined(LINALG_FIX_RANK_UPDATES)
      /* decltype(E) = */ void,
#endif // LINALG_FIX_RANK_UPDATES
      decltype(A),
      Triangle
    >::value;

  if constexpr (use_custom) {
    hermitian_matrix_rank_1_update(execpolicy_mapper(exec), alpha, x, A, t);
  }
  else {
    hermitian_matrix_rank_1_update(impl::inline_exec_t{}, alpha, x, A, t);
  }
}

#if defined(LINALG_FIX_RANK_UPDATES)
// Updating Hermitian rank-1 matrix update
// (ExecutionPolicy&&, alpha)
MDSPAN_TEMPLATE_REQUIRES(
  class ExecutionPolicy,
  class ScaleFactorType,
  class ElementType_x,
  class SizeType_x, ::std::size_t ext_x,
  class Layout_x,
  class Accessor_x,
  class ElementType_E,
  // Work-around for MDSPAN_TEMPLATE_REQUIRES not taking too many arguments:
  // combine (SizeType_E, numRows_E, numCols_E) into Extents_E
  // and add rank() == 2 constraint.
  class Extents_E,
  class Layout_E,
  class Accessor_E,
  class ElementType_A,
  class SizeType_A, ::std::size_t numRows_A,
  ::std::size_t numCols_A,
  class Layout_A, // TODO Implement Constraints and Mandates
  class Accessor_A,
  class Triangle,
  /* requires */ (
    impl::is_linalg_execution_policy_other_than_inline_v<ExecutionPolicy> &&
    (std::is_same_v<Triangle, lower_triangle_t> ||
     std::is_same_v<Triangle, upper_triangle_t>) &&
    Extents_E::rank() == 2
  )
)
void hermitian_matrix_rank_1_update(
  ExecutionPolicy&& exec,
  ScaleFactorType alpha,
  mdspan<ElementType_x, extents<SizeType_x, ext_x>, Layout_x, Accessor_x> x,
  mdspan<ElementType_E, Extents_E,                                 Layout_E, Accessor_E> E,
  mdspan<ElementType_A, extents<SizeType_A, numRows_A, numCols_A>, Layout_A, Accessor_A> A,
  Triangle t)
{
  constexpr bool use_custom = is_custom_hermitian_matrix_rank_1_update_avail<
      decltype(execpolicy_mapper(exec)),
      ScaleFactorType,
      decltype(x),
      decltype(E),
      decltype(A),
      Triangle
    >::value;

  if constexpr (use_custom) {
    hermitian_matrix_rank_1_update(execpolicy_mapper(exec), alpha, x, E, A, t);
  }
  else {
    hermitian_matrix_rank_1_update(impl::inline_exec_t{}, alpha, x, E, A, t);
  }
}
#endif // LINALG_FIX_RANK_UPDATES

// Overwriting Hermitian rank-1 matrix update
// (no execution policy, alpha)
MDSPAN_TEMPLATE_REQUIRES(
  class ScaleFactorType,
  class ElementType_x,
  class SizeType_x, ::std::size_t ext_x,
  class Layout_x,
  class Accessor_x,
  class ElementType_A,
  class SizeType_A, ::std::size_t numRows_A,
  ::std::size_t numCols_A,
  class Layout_A, // TODO Implement Constraints and Mandates
  class Accessor_A,
  class Triangle,
  /* requires */ (
    (! impl::is_linalg_execution_policy_other_than_inline_v<ScaleFactorType>) &&
    (std::is_same_v<Triangle, lower_triangle_t> ||
     std::is_same_v<Triangle, upper_triangle_t>)
  )
)
void hermitian_matrix_rank_1_update(
  ScaleFactorType alpha,
  mdspan<ElementType_x, extents<SizeType_x, ext_x>, Layout_x, Accessor_x> x,
  mdspan<ElementType_A, extents<SizeType_A, numRows_A, numCols_A>, Layout_A, Accessor_A> A,
  Triangle t)
{
  hermitian_matrix_rank_1_update(impl::default_exec_t{}, alpha, x, A, t);
}

#if defined(LINALG_FIX_RANK_UPDATES)
// Updating Hermitian rank-1 matrix update
// (no execution policy, alpha)
MDSPAN_TEMPLATE_REQUIRES(
  class ScaleFactorType,
  class ElementType_x,
  class SizeType_x, ::std::size_t ext_x,
  class Layout_x,
  class Accessor_x,
  class ElementType_E,
  class SizeType_E, ::std::size_t numRows_E,
  ::std::size_t numCols_E,
  class Layout_E,
  class Accessor_E,
  class ElementType_A,
  class SizeType_A, ::std::size_t numRows_A,
  ::std::size_t numCols_A,
  class Layout_A, // TODO Implement Constraints and Mandates
  class Accessor_A,
  class Triangle,
  /* requires */ (
    (! impl::is_linalg_execution_policy_other_than_inline_v<ScaleFactorType>) &&
    (std::is_same_v<Triangle, lower_triangle_t> ||
     std::is_same_v<Triangle, upper_triangle_t>)
  )
)
void hermitian_matrix_rank_1_update(
  ScaleFactorType alpha,
  mdspan<ElementType_x, extents<SizeType_x, ext_x>, Layout_x, Accessor_x> x,
  mdspan<ElementType_E, extents<SizeType_E, numRows_E, numCols_E>, Layout_E, Accessor_E> E,
  mdspan<ElementType_A, extents<SizeType_A, numRows_A, numCols_A>, Layout_A, Accessor_A> A,
  Triangle t)
{
  hermitian_matrix_rank_1_update(impl::default_exec_t{}, alpha, x, E, A, t);
}
#endif // LINALG_FIX_RANK_UPDATES

// Overwriting Hermitian rank-1 matrix update
// (inline_exec_t, no alpha)
MDSPAN_TEMPLATE_REQUIRES(
  class ElementType_x,
  class SizeType_x, ::std::size_t ext_x,
  class Layout_x,
  class Accessor_x,
  class ElementType_A,
  class SizeType_A, ::std::size_t numRows_A,
  ::std::size_t numCols_A,
  class Layout_A,
  class Accessor_A,
  class Triangle,
  /* requires */ (
    std::is_same_v<Triangle, lower_triangle_t> ||
    std::is_same_v<Triangle, upper_triangle_t>
  )
)
void hermitian_matrix_rank_1_update(
  impl::inline_exec_t&& /* exec */,
  mdspan<ElementType_x, extents<SizeType_x, ext_x>, Layout_x, Accessor_x> x,
  mdspan<ElementType_A, extents<SizeType_A, numRows_A, numCols_A>, Layout_A, Accessor_A> A,
  Triangle /* t */)
{
  using index_type = std::common_type_t<SizeType_x, SizeType_A>;

  if constexpr (std::is_same_v<Triangle, lower_triangle_t>) {
    for (index_type i = 0; i < A.extent(0); ++i) {
      for (index_type j = 0; j <= i; ++j) {
#if defined(LINALG_FIX_RANK_UPDATES)
        A(i,j) = x(i) * impl::conj_if_needed(x(j));
#else
        auto A_ij = i == j ? impl::real_if_needed(A(i,j)) : A(i,j);
        A(i,j) = A_ij + x(i) * impl::conj_if_needed(x(j));
#endif // LINALG_FIX_RANK_UPDATES
      }
    }
  }
  else { // upper triangle
    for (index_type j = 0; j < A.extent(1); ++j) {
      for (index_type i = 0; i <= j; ++i) {
#if defined(LINALG_FIX_RANK_UPDATES)
        A(i,j) = typename decltype(A)::value_type{};
#else
        if (i == j) {
          A(j,j) = impl::real_if_needed(A(j,j));
        }
#endif // LINALG_FIX_RANK_UPDATES
        A(i,j) += x(i) * impl::conj_if_needed(x(j));
      }
    }
  }
}

#if defined(LINALG_FIX_RANK_UPDATES)
// Updating Hermitian rank-1 matrix update
// (inline_exec_t, no alpha)
MDSPAN_TEMPLATE_REQUIRES(
  class ElementType_x,
  class SizeType_x, ::std::size_t ext_x,
  class Layout_x,
  class Accessor_x,
  class ElementType_E,
  class SizeType_E, ::std::size_t numRows_E,
  ::std::size_t numCols_E,
  class Layout_E,
  class Accessor_E,
  class ElementType_A,
  class SizeType_A, ::std::size_t numRows_A,
  ::std::size_t numCols_A,
  class Layout_A,
  class Accessor_A,
  class Triangle,
  /* requires */ (
    std::is_same_v<Triangle, lower_triangle_t> ||
    std::is_same_v<Triangle, upper_triangle_t>
  )
)
void hermitian_matrix_rank_1_update(
  impl::inline_exec_t&& /* exec */,
  mdspan<ElementType_x, extents<SizeType_x, ext_x>, Layout_x, Accessor_x> x,
  mdspan<ElementType_E, extents<SizeType_E, numRows_E, numCols_E>, Layout_E, Accessor_E> E,
  mdspan<ElementType_A, extents<SizeType_A, numRows_A, numCols_A>, Layout_A, Accessor_A> A,
  Triangle /* t */)
{
  using index_type = std::common_type_t<SizeType_x, SizeType_E, SizeType_A>;

  if constexpr (std::is_same_v<Triangle, lower_triangle_t>) {
    for (index_type i = 0; i < A.extent(0); ++i) {
      for (index_type j = 0; j <= i; ++j) {
        if (i == j) {
          A(i,j) = impl::real_if_needed(E(i,j)) + x(i) * impl::conj_if_needed(x(j));
        }
        else {
          A(i,j) = E(i,j) + x(i) * impl::conj_if_needed(x(j));
        }
      }
    }
  }
  else { // upper triangle
    for (index_type j = 0; j < A.extent(1); ++j) {
      for (index_type i = 0; i <= j; ++i) {
        if (i == j) {
          A(i,j) = impl::real_if_needed(E(i,j)) + x(i) * impl::conj_if_needed(x(j));
        }
        else {
          A(i,j) = E(i,j) + x(i) * impl::conj_if_needed(x(j));
        }
      }
    }
  }
}
#endif // LINALG_FIX_RANK_UPDATES

// Overwriting Hermitian rank-1 matrix update
// (ExecutionPolicy&&, no alpha)
MDSPAN_TEMPLATE_REQUIRES(
  class ExecutionPolicy,
  class ElementType_x,
  class SizeType_x, ::std::size_t ext_x,
  class Layout_x,
  class Accessor_x,
  class ElementType_A,
  class SizeType_A, ::std::size_t numRows_A,
  ::std::size_t numCols_A,
  class Layout_A,
  class Accessor_A,
  class Triangle,
  /* requires */ (
    impl::is_linalg_execution_policy_other_than_inline_v<ExecutionPolicy> &&
    (std::is_same_v<Triangle, lower_triangle_t> ||
     std::is_same_v<Triangle, upper_triangle_t>)
  )
)
void hermitian_matrix_rank_1_update(
  ExecutionPolicy&& exec,
  mdspan<ElementType_x, extents<SizeType_x, ext_x>, Layout_x, Accessor_x> x,
  mdspan<ElementType_A, extents<SizeType_A, numRows_A, numCols_A>, Layout_A, Accessor_A> A,
  Triangle t)
{
  constexpr bool use_custom = is_custom_hermitian_matrix_rank_1_update_avail<
      decltype(impl::map_execpolicy_with_check(exec)),
      /* ScaleFactorType = */ void,
      decltype(x),
#if defined(LINALG_FIX_RANK_UPDATES)
      /* decltype(E) = */ void,
#endif // LINALG_FIX_RANK_UPDATES
      decltype(A),
      Triangle
    >::value;

  if constexpr (use_custom) {
    hermitian_matrix_rank_1_update(impl::map_execpolicy_with_check(exec), x, A, t);
  }
  else {
    hermitian_matrix_rank_1_update(impl::inline_exec_t{}, x, A, t);
  }
}

#if defined(LINALG_FIX_RANK_UPDATES)
// Updating Hermitian rank-1 matrix update
// (ExecutionPolicy&&, no alpha)
MDSPAN_TEMPLATE_REQUIRES(
  class ExecutionPolicy,
  class ElementType_x,
  class SizeType_x, ::std::size_t ext_x,
  class Layout_x,
  class Accessor_x,
  class ElementType_E,
  class SizeType_E, ::std::size_t numRows_E,
  ::std::size_t numCols_E,
  class Layout_E,
  class Accessor_E,
  class ElementType_A,
  class SizeType_A, ::std::size_t numRows_A,
  ::std::size_t numCols_A,
  class Layout_A,
  class Accessor_A,
  class Triangle,
  /* requires */ (
    impl::is_linalg_execution_policy_other_than_inline_v<ExecutionPolicy> &&
    (std::is_same_v<Triangle, lower_triangle_t> ||
     std::is_same_v<Triangle, upper_triangle_t>)
  )
)
void hermitian_matrix_rank_1_update(
  ExecutionPolicy&& exec,
  mdspan<ElementType_x, extents<SizeType_x, ext_x>, Layout_x, Accessor_x> x,
  mdspan<ElementType_E, extents<SizeType_E, numRows_E, numCols_E>, Layout_E, Accessor_E> E,
  mdspan<ElementType_A, extents<SizeType_A, numRows_A, numCols_A>, Layout_A, Accessor_A> A,
  Triangle t)
{
  constexpr bool use_custom = is_custom_hermitian_matrix_rank_1_update_avail<
      decltype(impl::map_execpolicy_with_check(exec)),
      /* ScaleFactorType = */ void,
      decltype(x),
      decltype(E),
      decltype(A),
      Triangle
    >::value;

  if constexpr (use_custom) {
    hermitian_matrix_rank_1_update(impl::map_execpolicy_with_check(exec), x, E, A, t);
  }
  else {
    hermitian_matrix_rank_1_update(impl::inline_exec_t{}, x, E, A, t);
  }
}
#endif // LINALG_FIX_RANK_UPDATES

// Overwriting Hermitian rank-1 matrix update
// (no execution policy, no alpha)
MDSPAN_TEMPLATE_REQUIRES(
  class ElementType_x,
  class SizeType_x, ::std::size_t ext_x,
  class Layout_x,
  class Accessor_x,
  class ElementType_A,
  class SizeType_A, ::std::size_t numRows_A,
  ::std::size_t numCols_A,
  class Layout_A,
  class Accessor_A,
  class Triangle,
  /* requires */ (
    std::is_same_v<Triangle, lower_triangle_t> ||
    std::is_same_v<Triangle, upper_triangle_t>
  )
)
void hermitian_matrix_rank_1_update(
  mdspan<ElementType_x, extents<SizeType_x, ext_x>, Layout_x, Accessor_x> x,
  mdspan<ElementType_A, extents<SizeType_A, numRows_A, numCols_A>, Layout_A, Accessor_A> A,
  Triangle t)
{
  hermitian_matrix_rank_1_update(impl::default_exec_t{}, x, A, t);
}

#if defined(LINALG_FIX_RANK_UPDATES)
// Updating Hermitian rank-1 matrix update
// (no execution policy, no alpha)
MDSPAN_TEMPLATE_REQUIRES(
  class ElementType_x,
  class SizeType_x, ::std::size_t ext_x,
  class Layout_x,
  class Accessor_x,
  class ElementType_E,
  class SizeType_E, ::std::size_t numRows_E,
  ::std::size_t numCols_E,
  class Layout_E,
  class Accessor_E,
  class ElementType_A,
  class SizeType_A, ::std::size_t numRows_A,
  ::std::size_t numCols_A,
  class Layout_A,
  class Accessor_A,
  class Triangle,
  /* requires */ (
    std::is_same_v<Triangle, lower_triangle_t> ||
    std::is_same_v<Triangle, upper_triangle_t>
  )
)
void hermitian_matrix_rank_1_update(
  mdspan<ElementType_x, extents<SizeType_x, ext_x>, Layout_x, Accessor_x> x,
  mdspan<ElementType_E, extents<SizeType_E, numRows_E, numCols_E>, Layout_E, Accessor_E> E,
  mdspan<ElementType_A, extents<SizeType_A, numRows_A, numCols_A>, Layout_A, Accessor_A> A,
  Triangle t)
{
  hermitian_matrix_rank_1_update(impl::default_exec_t{}, x, E, A, t);
}
#endif // LINALG_FIX_RANK_UPDATES

} // end namespace linalg
} // end inline namespace __p1673_version_0
} // end namespace MDSPAN_IMPL_PROPOSED_NAMESPACE
} // end namespace MDSPAN_IMPL_STANDARD_NAMESPACE

#endif //LINALG_INCLUDE_EXPERIMENTAL___P1673_BITS_BLAS2_MATRIX_RANK_1_UPDATE_HPP_
