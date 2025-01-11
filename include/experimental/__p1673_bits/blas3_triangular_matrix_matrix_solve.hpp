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

#ifndef LINALG_INCLUDE_EXPERIMENTAL___P1673_BITS_BLAS3_TRIANGULAR_MATRIX_MATRIX_SOLVE_HPP_
#define LINALG_INCLUDE_EXPERIMENTAL___P1673_BITS_BLAS3_TRIANGULAR_MATRIX_MATRIX_SOLVE_HPP_

namespace MDSPAN_IMPL_STANDARD_NAMESPACE {
namespace MDSPAN_IMPL_PROPOSED_NAMESPACE {
inline namespace __p1673_version_0 {
namespace linalg {

namespace {

template<
  P1673_MATRIX_TEMPLATE_PARAMETERS( A ),
  class DiagonalStorage,
  P1673_MATRIX_TEMPLATE_PARAMETERS( B ),
  P1673_MATRIX_TEMPLATE_PARAMETERS( X )
>
void trsm_upper_triangular_left_side(
  P1673_MATRIX_PARAMETER( A ),
  DiagonalStorage d,
  P1673_MATRIX_PARAMETER( B ),
  P1673_MATRIX_PARAMETER( X ))
{
  constexpr bool explicit_diagonal =
    std::is_same_v<DiagonalStorage, explicit_diagonal_t>;
  using size_type = ::std::common_type_t<SizeType_A, SizeType_B, SizeType_X>;

  const size_type A_num_rows = A.extent(0);
  const size_type B_num_cols = B.extent(1);

  for (size_type k = 0; k < B_num_cols; ++k) {
    // One advantage of using signed index types is that you can write
    // descending loops with zero-based indices.
    // (AMK 6/8/21) i can't be a nonnegative type because the loop would be infinite
    for (ptrdiff_t i = A_num_rows - 1; i >= 0; --i) {
      // TODO this would be a great opportunity for an implementer to
      // add value, by accumulating in extended precision (or at least
      // in a type with the max precision of X and B).
      using sum_type = decltype (B(i,k) - A(0,0) * X(0,0));
      //using sum_type = typename out_object_t::element_type;
      sum_type t (B(i,k));
      for (size_type j = i + 1; j < A_num_rows; ++j) {
        t = t - A(i,j) * X(j,k);
      }
      if constexpr (explicit_diagonal) {
        X(i,k) = t / A(i,i);
      }
      else {
        X(i,k) = t;
      }
    }
  }
}

template<
  P1673_MATRIX_TEMPLATE_PARAMETERS( A ),
  class DiagonalStorage,
  P1673_MATRIX_TEMPLATE_PARAMETERS( B ),
  P1673_MATRIX_TEMPLATE_PARAMETERS( X )
>
void trsm_lower_triangular_left_side(
  P1673_MATRIX_PARAMETER( A ),
  DiagonalStorage d,
  P1673_MATRIX_PARAMETER( B ),
  P1673_MATRIX_PARAMETER( X ))
{
  constexpr bool explicit_diagonal =
    std::is_same_v<DiagonalStorage, explicit_diagonal_t>;
  using size_type = ::std::common_type_t<SizeType_A, SizeType_B, SizeType_X>;

  const size_type A_num_rows = A.extent(0);
  const size_type B_num_cols = B.extent(1);

  for (size_type k = 0; k < B_num_cols; ++k) {
    for (size_type i = 0; i < A_num_rows; ++i) {
      // TODO this would be a great opportunity for an implementer to
      // add value, by accumulating in extended precision (or at least
      // in a type with the max precision of X and B).
      ElementType_X t (B(i,k));
      for (size_type j = 0; j < i; ++j) {
        t = t - A(i,j) * X(j,k);
      }
      if constexpr (explicit_diagonal) {
        X(i,k) = t / A(i,i);
      }
      else {
        X(i,k) = t;
      }
    }
  }
}

template<
  P1673_MATRIX_TEMPLATE_PARAMETERS( A ),
  class DiagonalStorage,
  P1673_MATRIX_TEMPLATE_PARAMETERS( B ),
  P1673_MATRIX_TEMPLATE_PARAMETERS( X )
>
void trsm_upper_triangular_right_side(
  P1673_MATRIX_PARAMETER( A ),
  DiagonalStorage d,
  P1673_MATRIX_PARAMETER( B ),
  P1673_MATRIX_PARAMETER( X ))
{
  constexpr bool explicit_diagonal =
    std::is_same_v<DiagonalStorage, explicit_diagonal_t>;
  using size_type = ::std::common_type_t<SizeType_A, SizeType_B, SizeType_X>;

  const size_type B_num_rows = B.extent(0);
  const size_type A_num_cols = A.extent(1);

  for (size_type i = 0; i < B_num_rows; ++i) {
    for (size_type j = 0; j < A_num_cols; ++j) {
      using sum_type = decltype (B(i,j) - A(0,0) * X(0,0));
      sum_type t (B(i,j));
      for (size_type k = 0; k < j; ++k) {
        t = t - X(i,k) * A(k,j);
      }
      if constexpr (explicit_diagonal) {
        X(i,j) = t / A(j,j);
      }
      else {
        X(i,j) = t;
      }
    }
  }
}

template<
  P1673_MATRIX_TEMPLATE_PARAMETERS( A ),
  class DiagonalStorage,
  P1673_MATRIX_TEMPLATE_PARAMETERS( B ),
  P1673_MATRIX_TEMPLATE_PARAMETERS( X )
>
void trsm_lower_triangular_right_side(
  P1673_MATRIX_PARAMETER( A ),
  DiagonalStorage d,
  P1673_MATRIX_PARAMETER( B ),
  P1673_MATRIX_PARAMETER( X ))
{
  constexpr bool explicit_diagonal =
    std::is_same_v<DiagonalStorage, explicit_diagonal_t>;
  using size_type = ::std::common_type_t<SizeType_A, SizeType_B, SizeType_X>;
  using signed_index_type = ::std::make_signed_t<size_type>;

  const size_type B_num_rows = B.extent(0);
  const size_type A_num_rows = A.extent(0);
  const signed_index_type A_num_cols = A.extent(1);

  for (size_type i = 0; i < B_num_rows; ++i) {
    for (signed_index_type j = A_num_cols - 1; j >= 0; --j) {
      using sum_type = decltype (B(i,j) - A(0,0) * X(0,0));
      sum_type t (B(i,j));
      for (size_type k = j + 1; k < A_num_rows; ++k) {
        t = t - X(i,k) * A(k,j);
      }
      if constexpr (explicit_diagonal) {
        X(i,j) = t / A(j,j);
      }
      else {
        X(i,j) = t;
      }
    }
  }
}

template <class Exec, class A_t, class Tri_t, class D_t, class B_t, class X_t, class = void>
struct is_custom_tri_matrix_matrix_left_solve_avail : std::false_type {};

template <class Exec, class A_t, class Tri_t, class D_t, class B_t, class X_t>
struct is_custom_tri_matrix_matrix_left_solve_avail<
  Exec, A_t, Tri_t, D_t, B_t, X_t,
  std::enable_if_t<
    std::is_void_v<
      decltype(triangular_matrix_matrix_left_solve
	       (std::declval<Exec>(),
		std::declval<A_t>(),
		std::declval<Tri_t>(),
		std::declval<D_t>(),
		std::declval<B_t>(),
		std::declval<X_t>()
		)
	       )
      >
    && ! impl::is_inline_exec_v<Exec>
    >
  >
  : std::true_type{};

template <class Exec, class A_t, class Tri_t, class D_t, class B_t, class X_t, class = void>
struct is_custom_tri_matrix_matrix_right_solve_avail : std::false_type {};

template <class Exec, class A_t, class Tri_t, class D_t, class B_t, class X_t>
struct is_custom_tri_matrix_matrix_right_solve_avail<
  Exec, A_t, Tri_t, D_t, B_t, X_t,
  std::enable_if_t<
    std::is_void_v<
      decltype(triangular_matrix_matrix_right_solve
	       (std::declval<Exec>(),
		std::declval<A_t>(),
		std::declval<Tri_t>(),
		std::declval<D_t>(),
		std::declval<B_t>(),
		std::declval<X_t>()
		)
	       )
      >
    && ! impl::is_inline_exec_v<Exec>
    >
  >
  : std::true_type{};

template <class Exec, class A_t, class Tri_t, class D_t, class Side_t, class B_t, class X_t, class = void>
struct is_custom_tri_matrix_matrix_solve_avail : std::false_type {};

template <class Exec, class A_t, class Tri_t, class D_t, class Side_t, class B_t, class X_t>
struct is_custom_tri_matrix_matrix_solve_avail<
  Exec, A_t, Tri_t, D_t, Side_t, B_t, X_t,
  std::enable_if_t<
    std::is_void_v<
      decltype(triangular_matrix_matrix_right_solve
	       (std::declval<Exec>(),
		std::declval<A_t>(),
		std::declval<Tri_t>(),
		std::declval<D_t>(),
		std::declval<Side_t>(),
		std::declval<B_t>(),
		std::declval<X_t>()
		)
	       )
      >
    && ! impl::is_inline_exec_v<Exec>
    >
  >
  : std::true_type{};

} // end anonymous namespace

// triangular_matrix_matrix_left_solve

template<
  P1673_MATRIX_TEMPLATE_PARAMETERS( A ),
  class Triangle,
  class DiagonalStorage,
  P1673_MATRIX_TEMPLATE_PARAMETERS( B ),
  P1673_MATRIX_TEMPLATE_PARAMETERS( X )
>
void triangular_matrix_matrix_left_solve(
  impl::inline_exec_t&& /* exec */,
  P1673_MATRIX_PARAMETER( A ),
  Triangle /* t */,
  DiagonalStorage d,
  P1673_MATRIX_PARAMETER( B ),
  P1673_MATRIX_PARAMETER( X ))
{
  if (std::is_same_v<Triangle, lower_triangle_t>) {
    trsm_lower_triangular_left_side (A, d, B, X);
  }
  else {
    trsm_upper_triangular_left_side (A, d, B, X);
  }
}

template<
  class ExecutionPolicy,
  P1673_MATRIX_TEMPLATE_PARAMETERS( A ),
  class Triangle,
  class DiagonalStorage,
  P1673_MATRIX_TEMPLATE_PARAMETERS( B ),
  P1673_MATRIX_TEMPLATE_PARAMETERS( X )
>
void triangular_matrix_matrix_left_solve(
  ExecutionPolicy&& exec,
  P1673_MATRIX_PARAMETER( A ),
  Triangle t,
  DiagonalStorage d,
  P1673_MATRIX_PARAMETER( B ),
  P1673_MATRIX_PARAMETER( X ))
{
  constexpr bool use_custom = is_custom_tri_matrix_matrix_left_solve_avail<
    decltype(impl::map_execpolicy_with_check(exec)),
    decltype(A), Triangle, DiagonalStorage, decltype(B), decltype(X)>::value;

  if constexpr (use_custom) {
    triangular_matrix_matrix_left_solve(impl::map_execpolicy_with_check(exec), A, t, d, B, X);
  } else {
    triangular_matrix_matrix_left_solve(impl::inline_exec_t{}, A, t, d, B, X);
  }
}

template<
  P1673_MATRIX_TEMPLATE_PARAMETERS( A ),
  class Triangle,
  class DiagonalStorage,
  P1673_MATRIX_TEMPLATE_PARAMETERS( B ),
  P1673_MATRIX_TEMPLATE_PARAMETERS( X )
>
void triangular_matrix_matrix_left_solve(
  P1673_MATRIX_PARAMETER( A ),
  Triangle t,
  DiagonalStorage d,
  P1673_MATRIX_PARAMETER( B ),
  P1673_MATRIX_PARAMETER( X ))
{
  triangular_matrix_matrix_left_solve(impl::default_exec_t{}, A, t, d, B, X);
}

// triangular_matrix_matrix_right_solve

template<
  P1673_MATRIX_TEMPLATE_PARAMETERS( A ),
  class Triangle,
  class DiagonalStorage,
  P1673_MATRIX_TEMPLATE_PARAMETERS( B ),
  P1673_MATRIX_TEMPLATE_PARAMETERS( X )
>
void triangular_matrix_matrix_right_solve(
  impl::inline_exec_t&& /* exec */,
  P1673_MATRIX_PARAMETER( A ),
  Triangle /* t */,
  DiagonalStorage d,
  P1673_MATRIX_PARAMETER( B ),
  P1673_MATRIX_PARAMETER( X ))
{
  if (std::is_same_v<Triangle, lower_triangle_t>) {
    trsm_lower_triangular_right_side (A, d, B, X);
  }
  else {
    trsm_upper_triangular_right_side (A, d, B, X);
  }
}

template<
  class ExecutionPolicy,
  P1673_MATRIX_TEMPLATE_PARAMETERS( A ),
  class Triangle,
  class DiagonalStorage,
  P1673_MATRIX_TEMPLATE_PARAMETERS( B ),
  P1673_MATRIX_TEMPLATE_PARAMETERS( X )
>
void triangular_matrix_matrix_right_solve(
  ExecutionPolicy&& exec,
  P1673_MATRIX_PARAMETER( A ),
  Triangle t,
  DiagonalStorage d,
  P1673_MATRIX_PARAMETER( B ),
  P1673_MATRIX_PARAMETER( X ))
{
  constexpr bool use_custom = is_custom_tri_matrix_matrix_right_solve_avail<
    decltype(impl::map_execpolicy_with_check(exec)),
    decltype(A), Triangle, DiagonalStorage, decltype(B), decltype(X)>::value;

  if constexpr (use_custom) {
    triangular_matrix_matrix_right_solve(impl::map_execpolicy_with_check(exec), A, t, d, B, X);
  } else {
    triangular_matrix_matrix_right_solve(impl::inline_exec_t{}, A, t, d, B, X);
  }
}

template<
  P1673_MATRIX_TEMPLATE_PARAMETERS( A ),
  class Triangle,
  class DiagonalStorage,
  P1673_MATRIX_TEMPLATE_PARAMETERS( B ),
  P1673_MATRIX_TEMPLATE_PARAMETERS( X )
>
void triangular_matrix_matrix_right_solve(
  P1673_MATRIX_PARAMETER( A ),
  Triangle t,
  DiagonalStorage d,
  P1673_MATRIX_PARAMETER( B ),
  P1673_MATRIX_PARAMETER( X ))
{
  triangular_matrix_matrix_right_solve(impl::default_exec_t{}, A, t, d, B, X);
}

// triangular_matrix_matrix_solve

template<
  P1673_MATRIX_TEMPLATE_PARAMETERS( A ),
  class Triangle,
  class DiagonalStorage,
  class Side,
  P1673_MATRIX_TEMPLATE_PARAMETERS( B ),
  P1673_MATRIX_TEMPLATE_PARAMETERS( X )
>
void triangular_matrix_matrix_solve(
  impl::inline_exec_t&& /* exec */,
  P1673_MATRIX_PARAMETER( A ),
  Triangle t,
  DiagonalStorage d,
  Side /* s */,
  P1673_MATRIX_PARAMETER( B ),
  P1673_MATRIX_PARAMETER( X ))
{
  if constexpr (std::is_same_v<Side, left_side_t>) {
    triangular_matrix_matrix_left_solve(A, t, d, B, X);
  }
  else {
    triangular_matrix_matrix_right_solve(A, t, d, B, X);
  }
}

template<
  class ExecutionPolicy,
  P1673_MATRIX_TEMPLATE_PARAMETERS( A ),
  class Triangle,
  class DiagonalStorage,
  class Side,
  P1673_MATRIX_TEMPLATE_PARAMETERS( B ),
  P1673_MATRIX_TEMPLATE_PARAMETERS( X )
>
void triangular_matrix_matrix_solve(
  ExecutionPolicy&& exec ,
  P1673_MATRIX_PARAMETER( A ),
  Triangle t,
  DiagonalStorage d,
  Side s,
  P1673_MATRIX_PARAMETER( B ),
  P1673_MATRIX_PARAMETER( X ))
{
  constexpr bool use_custom = is_custom_tri_matrix_matrix_solve_avail<
    decltype(impl::map_execpolicy_with_check(exec)), decltype(A), Triangle,
    DiagonalStorage, Side, decltype(B), decltype(X)>::value;

  if constexpr (use_custom) {
    triangular_matrix_matrix_solve(impl::map_execpolicy_with_check(exec), A, t, d, s, B, X);
  } else {
    triangular_matrix_matrix_solve(impl::inline_exec_t{}, A, t, d, s, B, X);
  }
}

template<
  P1673_MATRIX_TEMPLATE_PARAMETERS( A ),
  class Triangle,
  class DiagonalStorage,
  class Side,
  P1673_MATRIX_TEMPLATE_PARAMETERS( B ),
  P1673_MATRIX_TEMPLATE_PARAMETERS( X )
>
void triangular_matrix_matrix_solve(
  P1673_MATRIX_PARAMETER( A ),
  Triangle t,
  DiagonalStorage d,
  Side s,
  P1673_MATRIX_PARAMETER( B ),
  P1673_MATRIX_PARAMETER( X ))
{
  triangular_matrix_matrix_solve(impl::default_exec_t{}, A, t, d, s, B, X);
}


} // end namespace linalg
} // end inline namespace __p1673_version_0
} // end namespace MDSPAN_IMPL_PROPOSED_NAMESPACE
} // end namespace MDSPAN_IMPL_STANDARD_NAMESPACE

#endif //LINALG_INCLUDE_EXPERIMENTAL___P1673_BITS_BLAS3_TRIANGULAR_MATRIX_MATRIX_SOLVE_HPP_
