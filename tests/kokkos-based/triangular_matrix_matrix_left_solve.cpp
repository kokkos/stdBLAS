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

#include "gtest_fixtures.hpp"
#include "helpers.hpp"

namespace{

using namespace kokkostesting;

template<class A_t,
         class X_t,
         class Triangle,
         class DiagonalStorage>
void triangular_matrix_matrix_left_solve_gold_solution(
        A_t A, Triangle t, DiagonalStorage d, X_t X)
{
  using size_type = typename std::experimental::extents<>::size_type;
  constexpr bool lower_triangle = std::is_same_v<
      Triangle, std::experimental::linalg::lower_triangle_t>;
  constexpr bool explicit_diagonal = std::is_same_v<
      DiagonalStorage, std::experimental::linalg::explicit_diagonal_t>;

  const size_type A_ext = A.extent(0); // = A.extent(1)
  const size_type num_vectors = X.extent(1);

  for (size_type k = 0; k < num_vectors; ++k) {
    for (size_type ii = 0; ii < A_ext; ++ii) {
      const size_type i = lower_triangle ? ii : A_ext - 1 - ii;
      // A(i, j) has lower triangle in i <= j
      const size_type j0 = lower_triangle ? 0 : i + 1;
      const size_type j1 = lower_triangle ? i : A_ext;
      for (size_type j = j0; j < j1; ++j) {
        X(i, k) -= A(i, j) * X(j, k);
      }
      if constexpr (explicit_diagonal) {
        X(i, k) /= A(i, i);
      }
    }
  }
}

template<class A_t,
         class B_t,
         class X_t,
         class Triangle,
         class DiagonalStorage>
void test_triangular_matrix_matrix_left_solve_impl(
        A_t A, B_t B, X_t X0, Triangle t, DiagonalStorage d)
{
  // copy x to leave original fixture intact
  auto x_data = create_stdvector_and_copy_rowwise(X0);
  auto X = make_mdspan(x_data.data(), X0.extent(0), X0.extent(1));

  const auto get_gold = [&](auto X_gold) {
      std::experimental::linalg::copy(B, X_gold);
      triangular_matrix_matrix_left_solve_gold_solution(A, t, d, X_gold);
    };
  const auto compute = [&]() {
      std::experimental::linalg::triangular_matrix_matrix_left_solve(
        KokkosKernelsSTD::kokkos_exec<>(), A, t, d, B, X);
    };
  const auto tol = tolerance<typename X_t::value_type>(1e-12, 1e-4f);
  test_op_CAB(A, B, X, tol, get_gold, compute);
}

} // anonymous namespace

#define DEFINE_TESTS(blas_val_type)                                              \
TEST_F(blas2_signed_##blas_val_type##_fixture,                                   \
       kokkos_triangular_matrix_matrix_left_solve) {                             \
  using val_t = typename blas2_signed_##blas_val_type##_fixture::value_type;     \
  run_checked_tests<val_t>("kokkos_", "triangular_matrix_matrix_left_solve", "", \
                           #blas_val_type, [&]() {                               \
                                                                                 \
    test_triangular_matrix_matrix_left_solve_impl(A_sym_e0, A_e0e1, B_e0e1,      \
                         std::experimental::linalg::lower_triangle,              \
                         std::experimental::linalg::implicit_unit_diagonal);     \
    test_triangular_matrix_matrix_left_solve_impl(A_sym_e0, A_e0e1, B_e0e1,      \
                         std::experimental::linalg::upper_triangle,              \
                         std::experimental::linalg::explicit_diagonal);          \
                                                                                 \
  });                                                                            \
}

DEFINE_TESTS(double)
DEFINE_TESTS(float)
DEFINE_TESTS(complex_double)