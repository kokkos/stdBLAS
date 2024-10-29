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

#ifndef LINALG_INCLUDE_EXPERIMENTAL___P1673_BITS_BLAS3_MATRIX_RANK_2K_UPDATE_HPP_
#define LINALG_INCLUDE_EXPERIMENTAL___P1673_BITS_BLAS3_MATRIX_RANK_2K_UPDATE_HPP_

namespace MDSPAN_IMPL_STANDARD_NAMESPACE {
namespace MDSPAN_IMPL_PROPOSED_NAMESPACE {
inline namespace __p1673_version_0 {
namespace linalg {

namespace{

#if defined(LINALG_FIX_RANK_UPDATES)

template <class Exec, class A_t, class B_t, class E_t, class C_t, class Tr_t, class = void>
struct is_custom_sym_mat_rank_2k_update_avail : std::false_type {};

template <class Exec, class A_t, class B_t, class C_t, class Tr_t>
struct is_custom_sym_mat_rank_2k_update_avail<
  Exec, A_t, B_t, void, C_t, Tr_t,
  std::enable_if_t<
    std::is_void_v<
      decltype(symmetric_matrix_rank_2k_update(std::declval<Exec>(),
					       std::declval<A_t>(),
					       std::declval<B_t>(),
					       std::declval<C_t>(),
					       std::declval<Tr_t>()))
      >
    && ! impl::is_inline_exec_v<Exec>
    >
  >
  : std::true_type{};

template <class Exec, class A_t, class B_t, class E_t, class C_t, class Tr_t>
struct is_custom_sym_mat_rank_2k_update_avail<
  Exec, A_t, B_t, E_t, C_t, Tr_t,
  std::enable_if_t<
    std::is_void_v<
      decltype(symmetric_matrix_rank_2k_update(std::declval<Exec>(),
					       std::declval<A_t>(),
					       std::declval<B_t>(),
					       std::declval<E_t>(),
    				       std::declval<C_t>(),
					       std::declval<Tr_t>()))
      >
    && ! impl::is_inline_exec_v<Exec>
    >
  >
  : std::true_type{};


#else

template <class Exec, class A_t, class B_t, class C_t, class Tr_t, class = void>
struct is_custom_sym_mat_rank_2k_update_avail : std::false_type {};

template <class Exec, class A_t, class B_t, class C_t, class Tr_t>
struct is_custom_sym_mat_rank_2k_update_avail<
  Exec, A_t, B_t, C_t, Tr_t,
  std::enable_if_t<
    std::is_void_v<
      decltype(symmetric_matrix_rank_2k_update(std::declval<Exec>(),
					       std::declval<A_t>(),
					       std::declval<B_t>(),
					       std::declval<C_t>(),
					       std::declval<Tr_t>()))
      >
    && ! impl::is_inline_exec_v<Exec>
    >
  >
  : std::true_type{};

#endif

#if defined(LINALG_FIX_RANK_UPDATES)

template <class Exec, class A_t, class B_t, class E_t, class C_t, class Tr_t, class = void>
struct is_custom_herm_mat_rank_2k_update_avail : std::false_type {};

template <class Exec, class A_t, class B_t, class C_t, class Tr_t>
struct is_custom_herm_mat_rank_2k_update_avail<
  Exec, A_t, B_t, void, C_t, Tr_t,
  std::enable_if_t<
    std::is_void_v<
      decltype(hermitian_matrix_rank_2k_update(std::declval<Exec>(),
					       std::declval<A_t>(),
					       std::declval<B_t>(),
					       std::declval<C_t>(),
					       std::declval<Tr_t>()))
      >
    && ! impl::is_inline_exec_v<Exec>
    >
  >
  : std::true_type{};

template <class Exec, class A_t, class B_t, class E_t, class C_t, class Tr_t>
struct is_custom_herm_mat_rank_2k_update_avail<
  Exec, A_t, B_t, E_t, C_t, Tr_t,
  std::enable_if_t<
    std::is_void_v<
      decltype(hermitian_matrix_rank_2k_update(std::declval<Exec>(),
					       std::declval<A_t>(),
					       std::declval<B_t>(),
					       std::declval<E_t>(),
    				       std::declval<C_t>(),
					       std::declval<Tr_t>()))
      >
    && ! impl::is_inline_exec_v<Exec>
    >
  >
  : std::true_type{};

#else

template <class Exec, class A_t, class B_t, class C_t, class Tr_t, class = void>
struct is_custom_herm_mat_rank_2k_update_avail : std::false_type {};

template <class Exec, class A_t, class B_t, class C_t, class Tr_t>
struct is_custom_herm_mat_rank_2k_update_avail<
  Exec, A_t, B_t, C_t, Tr_t,
  std::enable_if_t<
    std::is_void_v<
      decltype(hermitian_matrix_rank_2k_update(std::declval<Exec>(),
					       std::declval<A_t>(),
					       std::declval<B_t>(),
					       std::declval<C_t>(),
					       std::declval<Tr_t>()))
      >
    && ! impl::is_inline_exec_v<Exec>
    >
  >
  : std::true_type{};

#endif

} // end anonym namespace

// Rank-2k update of a symmetric matrix

template<class ElementType_A,
         class SizeType_A, ::std::size_t numRows_A, ::std::size_t numCols_A,
         class Layout_A,
         class Accessor_A,
         class ElementType_B,
         class SizeType_B, ::std::size_t numRows_B, ::std::size_t numCols_B,
         class Layout_B,
         class Accessor_B,
         class ElementType_C,
         class SizeType_C, ::std::size_t numRows_C, ::std::size_t numCols_C,
         class Layout_C,
         class Accessor_C,
         class Triangle>
void symmetric_matrix_rank_2k_update(
  impl::inline_exec_t&& /* exec */,
  mdspan<ElementType_A, extents<SizeType_A, numRows_A, numCols_A>, Layout_A, Accessor_A> A,
  mdspan<ElementType_B, extents<SizeType_B, numRows_B, numCols_B>, Layout_B, Accessor_B> B,
  mdspan<ElementType_C, extents<SizeType_C, numRows_C, numCols_C>, Layout_C, Accessor_C> C,
  Triangle /* t */)
{
  constexpr bool lower_tri =
    std::is_same_v<Triangle, lower_triangle_t>;
  using size_type = ::std::common_type_t<SizeType_A, SizeType_B, SizeType_C>;

  for (size_type j = 0; j < C.extent(1); ++j) {
    const size_type i_lower = lower_tri ? j : size_type(0);
    const size_type i_upper = lower_tri ? C.extent(0) : j+1;
    for (size_type i = i_lower; i < i_upper; ++i) {
#if defined(LINALG_FIX_RANK_UPDATES)
      C(i,j) = ElementType_C{};
#endif
      for (size_type k = 0; k < A.extent(1); ++k) {
        C(i,j) += A(i,k)*B(j,k) + B(i,k)*A(j,k);
      }
    }
  }
}

template<class ExecutionPolicy,
         class ElementType_A,
         class SizeType_A, ::std::size_t numRows_A, ::std::size_t numCols_A,
         class Layout_A,
         class Accessor_A,
         class ElementType_B,
         class SizeType_B, ::std::size_t numRows_B, ::std::size_t numCols_B,
         class Layout_B,
         class Accessor_B,
         class ElementType_C,
         class SizeType_C, ::std::size_t numRows_C, ::std::size_t numCols_C,
         class Layout_C,
         class Accessor_C,
         class Triangle>
void symmetric_matrix_rank_2k_update(
  ExecutionPolicy&& exec ,
  mdspan<ElementType_A, extents<SizeType_A, numRows_A, numCols_A>, Layout_A, Accessor_A> A,
  mdspan<ElementType_B, extents<SizeType_B, numRows_B, numCols_B>, Layout_B, Accessor_B> B,
  mdspan<ElementType_C, extents<SizeType_C, numRows_C, numCols_C>, Layout_C, Accessor_C> C,
  Triangle t)
{
  constexpr bool use_custom = is_custom_sym_mat_rank_2k_update_avail<
    decltype(impl::map_execpolicy_with_check(exec)), decltype(A), decltype(B),
#if defined(LINALG_FIX_RANK_UPDATES)
    void,
#endif
    decltype(C), Triangle
    >::value;

  if constexpr (use_custom) {
    symmetric_matrix_rank_2k_update(impl::map_execpolicy_with_check(exec), A, B, C, t);
  } else {
    symmetric_matrix_rank_2k_update(impl::inline_exec_t{}, A, B, C, t);
  }
}

template<class ElementType_A,
         class SizeType_A, ::std::size_t numRows_A, ::std::size_t numCols_A,
         class Layout_A,
         class Accessor_A,
         class ElementType_B,
         class SizeType_B, ::std::size_t numRows_B, ::std::size_t numCols_B,
         class Layout_B,
         class Accessor_B,
         class ElementType_C,
         class SizeType_C, ::std::size_t numRows_C, ::std::size_t numCols_C,
         class Layout_C,
         class Accessor_C,
         class Triangle>
void symmetric_matrix_rank_2k_update(
  mdspan<ElementType_A, extents<SizeType_A, numRows_A, numCols_A>, Layout_A, Accessor_A> A,
  mdspan<ElementType_B, extents<SizeType_B, numRows_B, numCols_B>, Layout_B, Accessor_B> B,
  mdspan<ElementType_C, extents<SizeType_C, numRows_C, numCols_C>, Layout_C, Accessor_C> C,
  Triangle t )
{
  symmetric_matrix_rank_2k_update(impl::default_exec_t{}, A, B, C, t);
}

#if defined(LINALG_FIX_RANK_UPDATES)

// Rank-2k update of a symmetric matrix

template<class ElementType_A,
         class SizeType_A, ::std::size_t numRows_A, ::std::size_t numCols_A,
         class Layout_A,
         class Accessor_A,
         class ElementType_B,
         class SizeType_B, ::std::size_t numRows_B, ::std::size_t numCols_B,
         class Layout_B,
         class Accessor_B,
         class ElementType_E,
         class SizeType_E, ::std::size_t numRows_E, ::std::size_t numCols_E,
         class Layout_E,
         class Accessor_E,
         class ElementType_C,
         class SizeType_C, ::std::size_t numRows_C, ::std::size_t numCols_C,
         class Layout_C,
         class Accessor_C,
         class Triangle>
void symmetric_matrix_rank_2k_update(
  impl::inline_exec_t&& /* exec */,
  mdspan<ElementType_A, extents<SizeType_A, numRows_A, numCols_A>, Layout_A, Accessor_A> A,
  mdspan<ElementType_B, extents<SizeType_B, numRows_B, numCols_B>, Layout_B, Accessor_B> B,
  mdspan<ElementType_E, extents<SizeType_E, numRows_E, numCols_E>, Layout_E, Accessor_E> E,
  mdspan<ElementType_C, extents<SizeType_C, numRows_C, numCols_C>, Layout_C, Accessor_C> C,
  Triangle /* t */)
{
  constexpr bool lower_tri =
    std::is_same_v<Triangle, lower_triangle_t>;
  using size_type = ::std::common_type_t<SizeType_A, SizeType_B, SizeType_C>;

  for (size_type j = 0; j < C.extent(1); ++j) {
    const size_type i_lower = lower_tri ? j : size_type(0);
    const size_type i_upper = lower_tri ? C.extent(0) : j+1;
    for (size_type i = i_lower; i < i_upper; ++i) {
      C(i,j) = E(i,j);
      for (size_type k = 0; k < A.extent(1); ++k) {
        C(i,j) += A(i,k)*B(j,k) + B(i,k)*A(j,k);
      }
    }
  }
}

template<class ExecutionPolicy,
         class ElementType_A,
         class SizeType_A, ::std::size_t numRows_A, ::std::size_t numCols_A,
         class Layout_A,
         class Accessor_A,
         class ElementType_B,
         class SizeType_B, ::std::size_t numRows_B, ::std::size_t numCols_B,
         class Layout_B,
         class Accessor_B,
         class ElementType_E,
         class SizeType_E, ::std::size_t numRows_E, ::std::size_t numCols_E,
         class Layout_E,
         class Accessor_E,
         class ElementType_C,
         class SizeType_C, ::std::size_t numRows_C, ::std::size_t numCols_C,
         class Layout_C,
         class Accessor_C,
         class Triangle>
void symmetric_matrix_rank_2k_update(
  ExecutionPolicy&& exec ,
  mdspan<ElementType_A, extents<SizeType_A, numRows_A, numCols_A>, Layout_A, Accessor_A> A,
  mdspan<ElementType_B, extents<SizeType_B, numRows_B, numCols_B>, Layout_B, Accessor_B> B,
  mdspan<ElementType_E, extents<SizeType_E, numRows_E, numCols_E>, Layout_E, Accessor_E> E,
  mdspan<ElementType_C, extents<SizeType_C, numRows_C, numCols_C>, Layout_C, Accessor_C> C,
  Triangle t)
{
  constexpr bool use_custom = is_custom_sym_mat_rank_2k_update_avail<
    decltype(impl::map_execpolicy_with_check(exec)), decltype(A), decltype(B), decltype(E), decltype(C), Triangle
    >::value;

  if constexpr (use_custom) {
    symmetric_matrix_rank_2k_update(impl::map_execpolicy_with_check(exec), A, B, E, C, t);
  } else {
    symmetric_matrix_rank_2k_update(impl::inline_exec_t{}, A, B, E, C, t);
  }
}

template<class ElementType_A,
         class SizeType_A, ::std::size_t numRows_A, ::std::size_t numCols_A,
         class Layout_A,
         class Accessor_A,
         class ElementType_B,
         class SizeType_B, ::std::size_t numRows_B, ::std::size_t numCols_B,
         class Layout_B,
         class Accessor_B,
         class ElementType_E,
         class SizeType_E, ::std::size_t numRows_E, ::std::size_t numCols_E,
         class Layout_E,
         class Accessor_E,
         class ElementType_C,
         class SizeType_C, ::std::size_t numRows_C, ::std::size_t numCols_C,
         class Layout_C,
         class Accessor_C,
         class Triangle>
void symmetric_matrix_rank_2k_update(
  mdspan<ElementType_A, extents<SizeType_A, numRows_A, numCols_A>, Layout_A, Accessor_A> A,
  mdspan<ElementType_B, extents<SizeType_B, numRows_B, numCols_B>, Layout_B, Accessor_B> B,
  mdspan<ElementType_E, extents<SizeType_E, numRows_E, numCols_E>, Layout_E, Accessor_E> E,
  mdspan<ElementType_C, extents<SizeType_C, numRows_C, numCols_C>, Layout_C, Accessor_C> C,
  Triangle t )
{
  symmetric_matrix_rank_2k_update(impl::default_exec_t{}, A, B, E, C, t);
}

#endif

// Overwriting Rank-2k update of a Hermitian matrix

template<class ElementType_A,
         class SizeType_A, ::std::size_t numRows_A, ::std::size_t numCols_A,
         class Layout_A,
         class Accessor_A,
         class ElementType_B,
         class SizeType_B, ::std::size_t numRows_B, ::std::size_t numCols_B,
         class Layout_B,
         class Accessor_B,
         class ElementType_C,
         class SizeType_C, ::std::size_t numRows_C, ::std::size_t numCols_C,
         class Layout_C,
         class Accessor_C,
         class Triangle>
void hermitian_matrix_rank_2k_update(
  impl::inline_exec_t&& /* exec */,
  mdspan<ElementType_A, extents<SizeType_A, numRows_A, numCols_A>, Layout_A, Accessor_A> A,
  mdspan<ElementType_B, extents<SizeType_B, numRows_B, numCols_B>, Layout_B, Accessor_B> B,
  mdspan<ElementType_C, extents<SizeType_C, numRows_C, numCols_C>, Layout_C, Accessor_C> C,
  Triangle /* t */)
{
  constexpr bool lower_tri =
    std::is_same_v<Triangle, lower_triangle_t>;
  using size_type = ::std::common_type_t<SizeType_A, SizeType_B, SizeType_C>;

  for (size_type j = 0; j < C.extent(1); ++j) {
    const size_type i_lower = lower_tri ? j : size_type(0);
    const size_type i_upper = lower_tri ? C.extent(0) : j+1;
#if !defined(LINALG_FIX_RANK_UPDATES)
    C(j,j) = impl::real_if_needed(C(j,j));
#endif
    for (size_type i = i_lower; i < i_upper; ++i) {
#if defined(LINALG_FIX_RANK_UPDATES)
      C(i,j) = ElementType_C{};
#endif
      for (size_type k = 0; k < A.extent(1); ++k) {
        C(i,j) += A(i,k) * impl::conj_if_needed(B(j,k)) + B(i,k) * impl::conj_if_needed(A(j,k));
      }
    }
  }
}

template<class ExecutionPolicy,
         class ElementType_A,
         class SizeType_A, ::std::size_t numRows_A, ::std::size_t numCols_A,
         class Layout_A,
         class Accessor_A,
         class ElementType_B,
         class SizeType_B, ::std::size_t numRows_B, ::std::size_t numCols_B,
         class Layout_B,
         class Accessor_B,
         class ElementType_C,
         class SizeType_C, ::std::size_t numRows_C, ::std::size_t numCols_C,
         class Layout_C,
         class Accessor_C,
         class Triangle>
void hermitian_matrix_rank_2k_update(
  ExecutionPolicy&& exec,
  mdspan<ElementType_A, extents<SizeType_A, numRows_A, numCols_A>, Layout_A, Accessor_A> A,
  mdspan<ElementType_B, extents<SizeType_B, numRows_B, numCols_B>, Layout_B, Accessor_B> B,
  mdspan<ElementType_C, extents<SizeType_C, numRows_C, numCols_C>, Layout_C, Accessor_C> C,
  Triangle t)
{
  constexpr bool use_custom = is_custom_herm_mat_rank_2k_update_avail<
    decltype(impl::map_execpolicy_with_check(exec)), decltype(A), decltype(B),
#if defined(LINALG_FIX_RANK_UPDATES)
    void,
#endif
    decltype(C), Triangle
    >::value;

  if constexpr (use_custom) {
    hermitian_matrix_rank_2k_update(impl::map_execpolicy_with_check(exec), A, B, C, t);
  } else {
    hermitian_matrix_rank_2k_update(impl::inline_exec_t{}, A, B, C, t);
  }
}

template<class ElementType_A,
         class SizeType_A, ::std::size_t numRows_A, ::std::size_t numCols_A,
         class Layout_A,
         class Accessor_A,
         class ElementType_B,
         class SizeType_B, ::std::size_t numRows_B, ::std::size_t numCols_B,
         class Layout_B,
         class Accessor_B,
         class ElementType_C,
         class SizeType_C, ::std::size_t numRows_C, ::std::size_t numCols_C,
         class Layout_C,
         class Accessor_C,
         class Triangle>
void hermitian_matrix_rank_2k_update(
  mdspan<ElementType_A, extents<SizeType_A, numRows_A, numCols_A>, Layout_A, Accessor_A> A,
  mdspan<ElementType_B, extents<SizeType_B, numRows_B, numCols_B>, Layout_B, Accessor_B> B,
  mdspan<ElementType_C, extents<SizeType_C, numRows_C, numCols_C>, Layout_C, Accessor_C> C,
  Triangle t )
{
  hermitian_matrix_rank_2k_update(impl::default_exec_t{}, A, B, C, t);
}

#if defined(LINALG_FIX_RANK_UPDATES)

// Rank-2k update of a Hermitian matrix

template<class ElementType_A,
         class SizeType_A, ::std::size_t numRows_A, ::std::size_t numCols_A,
         class Layout_A,
         class Accessor_A,
         class ElementType_B,
         class SizeType_B, ::std::size_t numRows_B, ::std::size_t numCols_B,
         class Layout_B,
         class Accessor_B,
         class ElementType_E,
         class SizeType_E, ::std::size_t numRows_E, ::std::size_t numCols_E,
         class Layout_E,
         class Accessor_E,
         class ElementType_C,
         class SizeType_C, ::std::size_t numRows_C, ::std::size_t numCols_C,
         class Layout_C,
         class Accessor_C,
         class Triangle>
void hermitian_matrix_rank_2k_update(
  impl::inline_exec_t&& /* exec */,
  mdspan<ElementType_A, extents<SizeType_A, numRows_A, numCols_A>, Layout_A, Accessor_A> A,
  mdspan<ElementType_B, extents<SizeType_B, numRows_B, numCols_B>, Layout_B, Accessor_B> B,
  mdspan<ElementType_E, extents<SizeType_E, numRows_E, numCols_E>, Layout_E, Accessor_E> E,
  mdspan<ElementType_C, extents<SizeType_C, numRows_C, numCols_C>, Layout_C, Accessor_C> C,
  Triangle /* t */)
{
  constexpr bool lower_tri =
    std::is_same_v<Triangle, lower_triangle_t>;
  using size_type = ::std::common_type_t<SizeType_A, SizeType_B, SizeType_C>;

  for (size_type j = 0; j < C.extent(1); ++j) {
    const size_type i_lower = lower_tri ? j : size_type(0);
    const size_type i_upper = lower_tri ? C.extent(0) : j+1;
    C(j,j) = impl::real_if_needed(E(j,j));
    for (size_type i = i_lower; i < i_upper; ++i) {
      for (size_type k = 0; k < A.extent(1); ++k) {
        C(i,j) += A(i,k) * impl::conj_if_needed(B(j,k)) + B(i,k) * impl::conj_if_needed(A(j,k));
      }
    }
  }
}

template<class ExecutionPolicy,
         class ElementType_A,
         class SizeType_A, ::std::size_t numRows_A, ::std::size_t numCols_A,
         class Layout_A,
         class Accessor_A,
         class ElementType_B,
         class SizeType_B, ::std::size_t numRows_B, ::std::size_t numCols_B,
         class Layout_B,
         class Accessor_B,
         class ElementType_E,
         class SizeType_E, ::std::size_t numRows_E, ::std::size_t numCols_E,
         class Layout_E,
         class Accessor_E,
         class ElementType_C,
         class SizeType_C, ::std::size_t numRows_C, ::std::size_t numCols_C,
         class Layout_C,
         class Accessor_C,
         class Triangle>
void hermitian_matrix_rank_2k_update(
  ExecutionPolicy&& exec,
  mdspan<ElementType_A, extents<SizeType_A, numRows_A, numCols_A>, Layout_A, Accessor_A> A,
  mdspan<ElementType_B, extents<SizeType_B, numRows_B, numCols_B>, Layout_B, Accessor_B> B,
  mdspan<ElementType_E, extents<SizeType_E, numRows_E, numCols_E>, Layout_E, Accessor_E> E,
  mdspan<ElementType_C, extents<SizeType_C, numRows_C, numCols_C>, Layout_C, Accessor_C> C,
  Triangle t)
{
  constexpr bool use_custom = is_custom_herm_mat_rank_2k_update_avail<
    decltype(impl::map_execpolicy_with_check(exec)), decltype(A), decltype(B), decltype(E), decltype(C), Triangle
    >::value;

  if constexpr (use_custom) {
    hermitian_matrix_rank_2k_update(impl::map_execpolicy_with_check(exec), A, B, E, C, t);
  } else {
    hermitian_matrix_rank_2k_update(impl::inline_exec_t{}, A, B, E, C, t);
  }
}

template<class ElementType_A,
         class SizeType_A, ::std::size_t numRows_A, ::std::size_t numCols_A,
         class Layout_A,
         class Accessor_A,
         class ElementType_B,
         class SizeType_B, ::std::size_t numRows_B, ::std::size_t numCols_B,
         class Layout_B,
         class Accessor_B,
         class ElementType_E,
         class SizeType_E, ::std::size_t numRows_E, ::std::size_t numCols_E,
         class Layout_E,
         class Accessor_E,
         class ElementType_C,
         class SizeType_C, ::std::size_t numRows_C, ::std::size_t numCols_C,
         class Layout_C,
         class Accessor_C,
         class Triangle>
void hermitian_matrix_rank_2k_update(
  mdspan<ElementType_A, extents<SizeType_A, numRows_A, numCols_A>, Layout_A, Accessor_A> A,
  mdspan<ElementType_B, extents<SizeType_B, numRows_B, numCols_B>, Layout_B, Accessor_B> B,
  mdspan<ElementType_E, extents<SizeType_E, numRows_E, numCols_E>, Layout_E, Accessor_E> E,
  mdspan<ElementType_C, extents<SizeType_C, numRows_C, numCols_C>, Layout_C, Accessor_C> C,
  Triangle t )
{
  hermitian_matrix_rank_2k_update(impl::default_exec_t{}, A, B, E, C, t);
}

#endif

} // end namespace linalg
} // end inline namespace __p1673_version_0
} // end namespace MDSPAN_IMPL_PROPOSED_NAMESPACE
} // end namespace MDSPAN_IMPL_STANDARD_NAMESPACE

#endif //LINALG_INCLUDE_EXPERIMENTAL___P1673_BITS_BLAS3_MATRIX_RANK_2K_UPDATE_HPP_
