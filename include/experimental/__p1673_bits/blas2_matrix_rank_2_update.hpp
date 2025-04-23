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

#ifndef LINALG_INCLUDE_EXPERIMENTAL___P1673_BITS_BLAS2_MATRIX_RANK_2_UPDATE_HPP_
#define LINALG_INCLUDE_EXPERIMENTAL___P1673_BITS_BLAS2_MATRIX_RANK_2_UPDATE_HPP_

namespace MDSPAN_IMPL_STANDARD_NAMESPACE {
namespace MDSPAN_IMPL_PROPOSED_NAMESPACE {
inline namespace __p1673_version_0 {
namespace linalg {

namespace {

#if defined(LINALG_FIX_RANK_UPDATES)

// For the overwriting case, use E_t = void.
template <class Exec, class x_t, class y_t, class E_t, class A_t, class Tr_t, class = void>
struct is_custom_symmetric_matrix_rank_2_update_avail : std::false_type {};

// Overwriting, ExecutionPolicy != inline_exec_t
template <class Exec, class x_t, class y_t, class A_t, class Tr_t>
struct is_custom_symmetric_matrix_rank_2_update_avail<
  Exec, x_t, y_t, /* E_t = */ void, A_t, Tr_t,
  std::enable_if_t<
    std::is_void_v<
      decltype(symmetric_matrix_rank_2_update
	       (std::declval<Exec>(),
		std::declval<x_t>(),
		std::declval<y_t>(),
		std::declval<A_t>(),
		std::declval<Tr_t>()
		)
	       )
      >
    && ! impl::is_inline_exec_v<Exec>
    >
  >
  : std::true_type{};

// Updating, ExecutionPolicy != inline_exec_t
template <class Exec, class x_t, class y_t, class E_t, class A_t, class Tr_t>
struct is_custom_symmetric_matrix_rank_2_update_avail<
  Exec, x_t, y_t, E_t, A_t, Tr_t,
  std::enable_if_t<
    std::is_void_v<
      decltype(symmetric_matrix_rank_2_update
	       (std::declval<Exec>(),
		std::declval<x_t>(),
		std::declval<y_t>(),
                std::declval<E_t>(), // implies not void
		std::declval<A_t>(),
		std::declval<Tr_t>()
		)
	       )
      >
    && ! impl::is_inline_exec_v<Exec>
    >
  >
  : std::true_type{};

#else

template <class Exec, class x_t, class y_t, class A_t, class Tr_t, class = void>
struct is_custom_symmetric_matrix_rank_2_update_avail : std::false_type {};

template <class Exec, class x_t, class y_t, class A_t, class Tr_t>
struct is_custom_symmetric_matrix_rank_2_update_avail<
  Exec, x_t, y_t, A_t, Tr_t,
  std::enable_if_t<
    std::is_void_v<
      decltype(symmetric_matrix_rank_2_update
	       (std::declval<Exec>(),
		std::declval<x_t>(),
		std::declval<y_t>(),
		std::declval<A_t>(),
		std::declval<Tr_t>()
		)
	       )
      >
    && ! impl::is_inline_exec_v<Exec>
    >
  >
  : std::true_type{};

#endif // LINALG_FIX_RANK_UPDATES

#if defined(LINALG_FIX_RANK_UPDATES)

// For the overwriting case, use E_t = void.
template <class Exec, class x_t, class y_t, class E_t, class A_t, class Tr_t, class = void>
struct is_custom_hermitian_matrix_rank_2_update_avail : std::false_type {};

// Overwriting, ExecutionPolicy != inline_exec_t
template <class Exec, class x_t, class y_t, class A_t, class Tr_t>
struct is_custom_hermitian_matrix_rank_2_update_avail<
  Exec, x_t, y_t, /* E_t = */ void, A_t, Tr_t,
  std::enable_if_t<
    std::is_void_v<
      decltype(hermitian_matrix_rank_2_update
	       (std::declval<Exec>(),
		std::declval<x_t>(),
		std::declval<y_t>(),
		std::declval<A_t>(),
		std::declval<Tr_t>()
		)
	       )
      >
    && ! impl::is_inline_exec_v<Exec>
    >
  >
  : std::true_type{};

template <class Exec, class x_t, class y_t, class E_t, class A_t, class Tr_t>
struct is_custom_hermitian_matrix_rank_2_update_avail<
  Exec, x_t, y_t, E_t, A_t, Tr_t,
  std::enable_if_t<
    std::is_void_v<
      decltype(hermitian_matrix_rank_2_update
	       (std::declval<Exec>(),
		std::declval<x_t>(),
		std::declval<y_t>(),
                std::declval<E_t>(), // implies not void
		std::declval<A_t>(),
		std::declval<Tr_t>()
		)
	       )
      >
    && ! impl::is_inline_exec_v<Exec>
    >
  >
  : std::true_type{};

#else

template <class Exec, class x_t, class y_t, class A_t, class Tr_t, class = void>
struct is_custom_hermitian_matrix_rank_2_update_avail : std::false_type {};

template <class Exec, class x_t, class y_t, class A_t, class Tr_t>
struct is_custom_hermitian_matrix_rank_2_update_avail<
  Exec, x_t, y_t, A_t, Tr_t,
  std::enable_if_t<
    std::is_void_v<
      decltype(hermitian_matrix_rank_2_update
	       (std::declval<Exec>(),
		std::declval<x_t>(),
		std::declval<y_t>(),
		std::declval<A_t>(),
		std::declval<Tr_t>()
		)
	       )
      >
    && ! impl::is_inline_exec_v<Exec>
    >
  >
  : std::true_type{};

#endif // LINALG_FIX_RANK_UPDATES

} // end anonymous namespace

// Symmetric matrix rank-2 update

// Overwriting symmetric rank-2 matrix update
// (inline_exec_t)
MDSPAN_TEMPLATE_REQUIRES(
  class ElementType_x,
  class IndexType_x, ::std::size_t ext_x,
  class Layout_x,
  class Accessor_x,
  class ElementType_y,
  class IndexType_y, ::std::size_t ext_y,
  class Layout_y,
  class Accessor_y,
  class ElementType_A,
  class IndexType_A, ::std::size_t numRows_A, ::std::size_t numCols_A,
  class Layout_A,
  class Accessor_A,
  class Triangle,
  /* requires */ (
    std::is_same_v<Triangle, lower_triangle_t> ||
    std::is_same_v<Triangle, upper_triangle_t>
  )
)  
void symmetric_matrix_rank_2_update(
  impl::inline_exec_t&& /* exec */,
  mdspan<ElementType_x, extents<IndexType_x, ext_x>, Layout_x, Accessor_x> x,
  mdspan<ElementType_y, extents<IndexType_y, ext_y>, Layout_y, Accessor_y> y,
  mdspan<ElementType_A, extents<IndexType_A, numRows_A, numCols_A>, Layout_A, Accessor_A> A,
  Triangle /* t */)
{
  using index_type = std::common_type_t<IndexType_x, IndexType_y, IndexType_A>;
  constexpr bool lower_tri = std::is_same_v<Triangle, lower_triangle_t>;
  for (index_type j = 0; j < A.extent(1); ++j) {
    const index_type i_lower = lower_tri ? j : index_type(0);
    const index_type i_upper = lower_tri ? A.extent(0) : j+1;
    for (index_type i = i_lower; i < i_upper; ++i) {
#if defined(LINALG_FIX_RANK_UPDATES)
      A(i,j) = x(i) * y(j) + y(i) * x(j);
#else
      A(i,j) += x(i) * y(j) + y(i) * x(j);
#endif // LINALG_FIX_RANK_UPDATES
    }
  }
}

#if defined(LINALG_FIX_RANK_UPDATES)
// Updating symmetric rank-2 matrix update
// (inline_exec_t)
MDSPAN_TEMPLATE_REQUIRES(
  class ElementType_x,
  class IndexType_x, ::std::size_t ext_x,
  class Layout_x,
  class Accessor_x,
  class ElementType_y,
  class IndexType_y, ::std::size_t ext_y,
  class Layout_y,
  class Accessor_y,
  class ElementType_E,
  class Extents_E,
  class Layout_E,
  class Accessor_E,
  class ElementType_A,
  class Extents_A,
  class Layout_A,
  class Accessor_A,
  class Triangle,
  /* requires */ (
    (std::is_same_v<Triangle, lower_triangle_t> ||
     std::is_same_v<Triangle, upper_triangle_t>) &&
    Extents_E::rank() == 2 &&
    Extents_A::rank() == 2
  )
)  
void symmetric_matrix_rank_2_update(
  impl::inline_exec_t&& /* exec */,
  mdspan<ElementType_x, extents<IndexType_x, ext_x>, Layout_x, Accessor_x> x,
  mdspan<ElementType_y, extents<IndexType_y, ext_y>, Layout_y, Accessor_y> y,
  mdspan<ElementType_E, Extents_E, Layout_E, Accessor_E> E,
  mdspan<ElementType_A, Extents_A, Layout_A, Accessor_A> A,
  Triangle /* t */)
{
  using index_type = std::common_type_t<IndexType_x, IndexType_y,
    typename Extents_E::index_type, typename Extents_A::index_type>;
  constexpr bool lower_tri = std::is_same_v<Triangle, lower_triangle_t>;
  for (index_type j = 0; j < A.extent(1); ++j) {
    const index_type i_lower = lower_tri ? j : index_type(0);
    const index_type i_upper = lower_tri ? A.extent(0) : j+1;

    for (index_type i = i_lower; i < i_upper; ++i) {
      A(i,j) = E(i,j) + x(i) * y(j) + y(i) * x(j);
    }
  }
}
#endif // LINALG_FIX_RANK_UPDATES

// Overwriting symmetric rank-2 matrix update
// (ExecutionPolicy&&)
MDSPAN_TEMPLATE_REQUIRES(
  class ExecutionPolicy,
  class ElementType_x,
  class IndexType_x, ::std::size_t ext_x,
  class Layout_x,
  class Accessor_x,
  class ElementType_y,
  class IndexType_y, ::std::size_t ext_y,
  class Layout_y,
  class Accessor_y,
  class ElementType_A,
  class Extents_A,
  class Layout_A,
  class Accessor_A,
  class Triangle,
  /* requires */ (
    impl::is_linalg_execution_policy_other_than_inline_v<ExecutionPolicy> &&
    (std::is_same_v<Triangle, lower_triangle_t> ||
     std::is_same_v<Triangle, upper_triangle_t>) &&
    Extents_A::rank() == 2
  )
)
void symmetric_matrix_rank_2_update(
  ExecutionPolicy&& exec,
  mdspan<ElementType_x, extents<IndexType_x, ext_x>, Layout_x, Accessor_x> x,
  mdspan<ElementType_y, extents<IndexType_y, ext_y>, Layout_y, Accessor_y> y,
  mdspan<ElementType_A, Extents_A, Layout_A, Accessor_A> A,
  Triangle t)
{
  constexpr bool use_custom = is_custom_symmetric_matrix_rank_2_update_avail<
      decltype(impl::map_execpolicy_with_check(exec)),
      decltype(x), decltype(y),
#if defined(LINALG_FIX_RANK_UPDATES)
      /* decltype(E) = */ void,
#endif // LINALG_FIX_RANK_UPDATES
      decltype(A), Triangle
    >::value;

  if constexpr (use_custom) {
    symmetric_matrix_rank_2_update(impl::map_execpolicy_with_check(exec), x, y, A, t);
  }
  else {
    symmetric_matrix_rank_2_update(impl::inline_exec_t{}, x, y, A, t);
  }
}

#if defined(LINALG_FIX_RANK_UPDATES)
// Updating symmetric rank-2 matrix update
// (ExecutionPolicy&&)
MDSPAN_TEMPLATE_REQUIRES(
  class ExecutionPolicy,
  class ElementType_x,
  class Extents_x,
  class Layout_x,
  class Accessor_x,
  class ElementType_y,
  class Extents_y,
  class Layout_y,
  class Accessor_y,
  class ElementType_E,
  class Extents_E,
  class Layout_E,
  class Accessor_E,
  class ElementType_A,
  class Extents_A,
  class Layout_A,
  class Accessor_A,
  class Triangle,
  /* requires */ (
    impl::is_linalg_execution_policy_other_than_inline_v<ExecutionPolicy> &&
    (std::is_same_v<Triangle, lower_triangle_t> ||
     std::is_same_v<Triangle, upper_triangle_t>) &&
    Extents_x::rank() == 1 &&
    Extents_y::rank() == 1 &&
    Extents_E::rank() == 2 &&
    Extents_A::rank() == 2
  )
)
void symmetric_matrix_rank_2_update(
  ExecutionPolicy&& exec,
  mdspan<ElementType_x, Extents_x, Layout_x, Accessor_x> x,
  mdspan<ElementType_y, Extents_y, Layout_y, Accessor_y> y,
  mdspan<ElementType_E, Extents_E, Layout_E, Accessor_E> E,
  mdspan<ElementType_A, Extents_A, Layout_A, Accessor_A> A,
  Triangle t)
{
  constexpr bool use_custom = is_custom_symmetric_matrix_rank_2_update_avail<
      decltype(impl::map_execpolicy_with_check(exec)),
      decltype(x), decltype(y),
      decltype(E),
      decltype(A), Triangle
    >::value;

  if constexpr (use_custom) {
    symmetric_matrix_rank_2_update(impl::map_execpolicy_with_check(exec), x, y, E, A, t);
  }
  else {
    symmetric_matrix_rank_2_update(impl::inline_exec_t{}, x, y, E, A, t);
  }
}
#endif // LINALG_FIX_RANK_UPDATES

// Overwriting symmetric rank-2 matrix update
// (No ExecutionPolicy&&)
MDSPAN_TEMPLATE_REQUIRES(
  class ElementType_x,
  class Extents_x,
  class Layout_x,
  class Accessor_x,
  class ElementType_y,
  class Extents_y,
  class Layout_y,
  class Accessor_y,
  class ElementType_A,
  class Extents_A,
  class Layout_A,
  class Accessor_A,
  class Triangle,
  /* requires */ (
    (std::is_same_v<Triangle, lower_triangle_t> ||
     std::is_same_v<Triangle, upper_triangle_t>) &&
    Extents_x::rank() == 1 &&
    Extents_y::rank() == 1 &&
    Extents_A::rank() == 2
  )
)
void symmetric_matrix_rank_2_update(
  mdspan<ElementType_x, Extents_x, Layout_x, Accessor_x> x,
  mdspan<ElementType_y, Extents_y, Layout_y, Accessor_y> y,
  mdspan<ElementType_A, Extents_A, Layout_A, Accessor_A> A,
  Triangle t)
{
  symmetric_matrix_rank_2_update(impl::default_exec_t{}, x, y, A, t);
}

#if defined(LINALG_FIX_RANK_UPDATES)
// Updating symmetric rank-2 matrix update
// (No ExecutionPolicy&&)
MDSPAN_TEMPLATE_REQUIRES(
  class ElementType_x,
  class Extents_x,
  class Layout_x,
  class Accessor_x,
  class ElementType_y,
  class Extents_y,
  class Layout_y,
  class Accessor_y,
  class ElementType_E,
  class Extents_E,
  class Layout_E,
  class Accessor_E,
  class ElementType_A,
  class Extents_A,
  class Layout_A,
  class Accessor_A,
  class Triangle,
  /* requires */ (
    (std::is_same_v<Triangle, lower_triangle_t> ||
     std::is_same_v<Triangle, upper_triangle_t>) &&
    Extents_x::rank() == 1 &&
    Extents_y::rank() == 1 &&
    Extents_E::rank() == 2 &&
    Extents_A::rank() == 2
  )
)
void symmetric_matrix_rank_2_update(
  mdspan<ElementType_x, Extents_x, Layout_x, Accessor_x> x,
  mdspan<ElementType_y, Extents_y, Layout_y, Accessor_y> y,
  mdspan<ElementType_E, Extents_E, Layout_E, Accessor_E> E,
  mdspan<ElementType_A, Extents_A, Layout_A, Accessor_A> A,
  Triangle t)
{
  symmetric_matrix_rank_2_update(impl::default_exec_t{}, x, y, E, A, t);
}
#endif // LINALG_FIX_RANK_UPDATES

// Hermitian matrix rank-2 update

// Overwriting Hermitian rank-2 matrix update
// (inline_exec_t)
MDSPAN_TEMPLATE_REQUIRES(
  class ElementType_x,
  class IndexType_x, ::std::size_t ext_x,
  class Layout_x,
  class Accessor_x,
  class ElementType_y,
  class IndexType_y, ::std::size_t ext_y,
  class Layout_y,
  class Accessor_y,
  class ElementType_A,
  class IndexType_A, ::std::size_t numRows_A, ::std::size_t numCols_A,
  class Layout_A,
  class Accessor_A,
  class Triangle,
  /* requires */ (
    std::is_same_v<Triangle, lower_triangle_t> ||
    std::is_same_v<Triangle, upper_triangle_t>
  )
)
void hermitian_matrix_rank_2_update(
  impl::inline_exec_t&& /* exec */,
  mdspan<ElementType_x, extents<IndexType_x, ext_x>, Layout_x, Accessor_x> x,
  mdspan<ElementType_y, extents<IndexType_y, ext_y>, Layout_y, Accessor_y> y,
  mdspan<ElementType_A, extents<IndexType_A, numRows_A, numCols_A>, Layout_A, Accessor_A> A,
  Triangle /* t */)
{
  using index_type = std::common_type_t<IndexType_x, IndexType_y, IndexType_A>;

  constexpr bool lower_tri =
    std::is_same_v<Triangle, lower_triangle_t>;
  for (index_type j = 0; j < A.extent(1); ++j) {
    const index_type i_lower = lower_tri ? j : index_type(0);
    const index_type i_upper = lower_tri ? A.extent(0) : j+1;

    A(j,j) = impl::real_if_needed(A(j,j));
    for (index_type i = i_lower; i < i_upper; ++i) {
#if defined(LINALG_FIX_RANK_UPDATES)
      A(i,j) = x(i) * impl::conj_if_needed(y(j)) + y(i) * impl::conj_if_needed(x(j));
#else
      A(i,j) += x(i) * impl::conj_if_needed(y(j)) + y(i) * impl::conj_if_needed(x(j));
#endif
    }
  }
}

#if defined(LINALG_FIX_RANK_UPDATES)
// Updating Hermitian rank-2 matrix update
// (inline_exec_t)
MDSPAN_TEMPLATE_REQUIRES(
  class ElementType_x,
  class IndexType_x, ::std::size_t ext_x,
  class Layout_x,
  class Accessor_x,
  class ElementType_y,
  class IndexType_y, ::std::size_t ext_y,
  class Layout_y,
  class Accessor_y,
  class ElementType_E,
  class Extents_E,
  class Layout_E,
  class Accessor_E,
  class ElementType_A,
  class Extents_A,
  class Layout_A,
  class Accessor_A,
  class Triangle,
  /* requires */ (
    (std::is_same_v<Triangle, lower_triangle_t> ||
     std::is_same_v<Triangle, upper_triangle_t>) &&
    Extents_E::rank() == 2 &&
    Extents_A::rank() == 2
  )
)
void hermitian_matrix_rank_2_update(
  impl::inline_exec_t&& /* exec */,
  mdspan<ElementType_x, extents<IndexType_x, ext_x>, Layout_x, Accessor_x> x,
  mdspan<ElementType_y, extents<IndexType_y, ext_y>, Layout_y, Accessor_y> y,
  mdspan<ElementType_E, Extents_E, Layout_E, Accessor_E> E,
  mdspan<ElementType_A, Extents_A, Layout_A, Accessor_A> A,  
  Triangle /* t */)
{
  using index_type = std::common_type_t<IndexType_x, IndexType_y,
    typename Extents_E::index_type, typename Extents_A::index_type>;
  constexpr bool lower_tri =
    std::is_same_v<Triangle, lower_triangle_t>;
  for (index_type j = 0; j < A.extent(1); ++j) {
    const index_type i_lower = lower_tri ? j : index_type(0);
    const index_type i_upper = lower_tri ? A.extent(0) : j+1;

    for (index_type i = i_lower; i < i_upper; ++i) {
      if (i == j) {
        A(i,j) = impl::real_if_needed(E(i,j)) +
          x(i) * impl::conj_if_needed(y(j)) + y(i) * impl::conj_if_needed(x(j));
      }
      else {
        A(i,j) = E(i,j) +
          x(i) * impl::conj_if_needed(y(j)) + y(i) * impl::conj_if_needed(x(j));
      }
    }
  }
}
#endif // LINALG_FIX_RANK_UPDATES

// Overwriting Hermitian rank-2 matrix update
// (ExecutionPolicy&&)
MDSPAN_TEMPLATE_REQUIRES(
  class ExecutionPolicy,
  class ElementType_x,
  class IndexType_x, ::std::size_t ext_x,
  class Layout_x,
  class Accessor_x,
  class ElementType_y,
  class IndexType_y, ::std::size_t ext_y,
  class Layout_y,
  class Accessor_y,
  class ElementType_A,
  class Extents_A,
  class Layout_A,
  class Accessor_A,
  class Triangle,
  /* requires */ (
    impl::is_linalg_execution_policy_other_than_inline_v<ExecutionPolicy> &&
    (std::is_same_v<Triangle, lower_triangle_t> ||
     std::is_same_v<Triangle, upper_triangle_t>) &&
    Extents_A::rank() == 2
  )
)
void hermitian_matrix_rank_2_update(
  ExecutionPolicy&& exec,
  mdspan<ElementType_x, extents<IndexType_x, ext_x>, Layout_x, Accessor_x> x,
  mdspan<ElementType_y, extents<IndexType_y, ext_y>, Layout_y, Accessor_y> y,
  mdspan<ElementType_A, Extents_A, Layout_A, Accessor_A> A,
  Triangle t)
{
  constexpr bool use_custom = is_custom_hermitian_matrix_rank_2_update_avail<
      decltype(impl::map_execpolicy_with_check(exec)),
      decltype(x), decltype(y),
#if defined(LINALG_FIX_RANK_UPDATES)
      /* decltype(E) = */ void,
#endif // LINALG_FIX_RANK_UPDATES
      decltype(A), Triangle
    >::value;

  if constexpr (use_custom) {
    hermitian_matrix_rank_2_update(impl::map_execpolicy_with_check(exec), x, y, A, t);
  }
  else {
    hermitian_matrix_rank_2_update(impl::inline_exec_t{}, x, y, A, t);
  }
}

#if defined(LINALG_FIX_RANK_UPDATES)
// Updating Hermitian rank-2 matrix update
// (ExecutionPolicy&&)
MDSPAN_TEMPLATE_REQUIRES(
  class ExecutionPolicy,
  class ElementType_x,
  class Extents_x,
  class Layout_x,
  class Accessor_x,
  class ElementType_y,
  class Extents_y,
  class Layout_y,
  class Accessor_y,
  class ElementType_E,
  class Extents_E,
  class Layout_E,
  class Accessor_E,
  class ElementType_A,
  class Extents_A,
  class Layout_A,
  class Accessor_A,
  class Triangle,
  /* requires */ (
    impl::is_linalg_execution_policy_other_than_inline_v<ExecutionPolicy> &&
    (std::is_same_v<Triangle, lower_triangle_t> ||
     std::is_same_v<Triangle, upper_triangle_t>) &&
    Extents_x::rank() == 1 &&
    Extents_y::rank() == 1 &&
    Extents_E::rank() == 2 &&
    Extents_A::rank() == 2
  )
)
void hermitian_matrix_rank_2_update(
  ExecutionPolicy&& exec,
  mdspan<ElementType_x, Extents_x, Layout_x, Accessor_x> x,
  mdspan<ElementType_y, Extents_y, Layout_y, Accessor_y> y,
  mdspan<ElementType_E, Extents_E, Layout_E, Accessor_E> E,
  mdspan<ElementType_A, Extents_A, Layout_A, Accessor_A> A,  
  Triangle t)
{
  constexpr bool use_custom = is_custom_hermitian_matrix_rank_2_update_avail<
      decltype(impl::map_execpolicy_with_check(exec)),
      decltype(x), decltype(y),
      decltype(E),
      decltype(A), Triangle
    >::value;

  if constexpr (use_custom) {
    hermitian_matrix_rank_2_update(impl::map_execpolicy_with_check(exec), x, y, E, A, t);
  }
  else {
    hermitian_matrix_rank_2_update(impl::inline_exec_t{}, x, y, E, A, t);
  }
}
#endif

// Overwriting symmetric rank-2 matrix update
// (No ExecutionPolicy&&)
MDSPAN_TEMPLATE_REQUIRES(
  class ElementType_x,
  class Extents_x,
  class Layout_x,
  class Accessor_x,
  class ElementType_y,
  class Extents_y,
  class Layout_y,
  class Accessor_y,
  class ElementType_A,
  class Extents_A,
  class Layout_A,
  class Accessor_A,
  class Triangle,
  /* requires */ (
    (std::is_same_v<Triangle, lower_triangle_t> ||
     std::is_same_v<Triangle, upper_triangle_t>) &&
    Extents_x::rank() == 1 &&
    Extents_y::rank() == 1 &&
    Extents_A::rank() == 2
  )
)
void hermitian_matrix_rank_2_update(
  mdspan<ElementType_x, Extents_x, Layout_x, Accessor_x> x,
  mdspan<ElementType_y, Extents_y, Layout_y, Accessor_y> y,
  mdspan<ElementType_A, Extents_A, Layout_A, Accessor_A> A,
  Triangle t)
{
  hermitian_matrix_rank_2_update(impl::default_exec_t{}, x, y, A, t);
}

#if defined(LINALG_FIX_RANK_UPDATES)
// Updating Hermitian rank-2 matrix update
// (No ExecutionPolicy&&)
MDSPAN_TEMPLATE_REQUIRES(
  class ElementType_x,
  class Extents_x,
  class Layout_x,
  class Accessor_x,
  class ElementType_y,
  class Extents_y,
  class Layout_y,
  class Accessor_y,
  class ElementType_E,
  class Extents_E,
  class Layout_E,
  class Accessor_E,
  class ElementType_A,
  class Extents_A,
  class Layout_A,
  class Accessor_A,
  class Triangle,
  /* requires */ (
    (std::is_same_v<Triangle, lower_triangle_t> ||
     std::is_same_v<Triangle, upper_triangle_t>) &&
    Extents_x::rank() == 1 &&
    Extents_y::rank() == 1 &&
    Extents_E::rank() == 2 &&
    Extents_A::rank() == 2
  )
)
void hermitian_matrix_rank_2_update(
  mdspan<ElementType_x, Extents_x, Layout_x, Accessor_x> x,
  mdspan<ElementType_y, Extents_y, Layout_y, Accessor_y> y,
  mdspan<ElementType_E, Extents_E, Layout_E, Accessor_E> E,
  mdspan<ElementType_A, Extents_A, Layout_A, Accessor_A> A,
  Triangle t)
{
  hermitian_matrix_rank_2_update(impl::default_exec_t{}, x, y, E, A, t);
}
#endif // LINALG_FIX_RANK_UPDATES

} // end namespace linalg
} // end inline namespace __p1673_version_0
} // end namespace MDSPAN_IMPL_PROPOSED_NAMESPACE
} // end namespace MDSPAN_IMPL_STANDARD_NAMESPACE

#endif //LINALG_INCLUDE_EXPERIMENTAL___P1673_BITS_BLAS2_MATRIX_RANK_2_UPDATE_HPP_
