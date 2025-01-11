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
         class x_t,
         class Triangle,
         class DiagonalStorage>
void triangular_matrix_vector_solve_gold_solution(A_t A, Triangle t, DiagonalStorage d, x_t x)
{
  using size_type = typename std::experimental::extents<>::size_type;
  constexpr bool lower_triangle = std::is_same_v<
      Triangle, std::experimental::linalg::lower_triangle_t>;
  constexpr bool explicit_diagonal = std::is_same_v<
      DiagonalStorage, std::experimental::linalg::explicit_diagonal_t>;

  const size_type ext0 = A.extent(0);
  const size_type ext1 = A.extent(1);

  for (size_type ii = 0; ii < ext0; ++ii) {
    const size_type i = lower_triangle ? ii : ext0 - 1 - ii;
    const size_type j0 = lower_triangle ? 0 : i + 1;
    const size_type j1 = lower_triangle ? i : ext1;
    for (size_type j = j0; j < j1; ++j) {
      x(i) -= A(i, j) * x(j);
    }
    if constexpr (explicit_diagonal) {
      x(i) /= A(i, i);
    }
  }
}

template<class A_t,
         class b_t,
         class x_t,
         class Triangle,
         class DiagonalStorage>
void test_triangular_matrix_vector_solve_impl(A_t A, b_t b, x_t x0, Triangle t, DiagonalStorage d)
{
  // copy x to leave original fixture intact
  auto x_data = create_stdvector_and_copy(x0);
  auto x = make_mdspan(x_data);

  const auto get_gold = [&](auto x_gold) {
    std::experimental::linalg::copy(b, x_gold);
    triangular_matrix_vector_solve_gold_solution(A, t, d, x_gold);
  };
  const auto compute = [&]() {
      std::experimental::linalg::triangular_matrix_vector_solve(
        KokkosKernelsSTD::kokkos_exec<>(), A, t, d, b, x);
    };
  const auto tol = tolerance<typename x_t::value_type>(1e-12, 1e-4f);
  test_op_xAb(A, b, x, tol, get_gold, compute);
}

} // anonymous namespace

#define DEFINE_TESTS(blas_val_type)                                          \
TEST_F(blas2_signed_##blas_val_type##_fixture,                               \
       kokkos_triangular_matrix_vector_solve) {                              \
  using val_t = typename blas2_signed_##blas_val_type##_fixture::value_type; \
  run_checked_tests<val_t>("kokkos_", "triangular_matrix_vector_solve", "",  \
                           #blas_val_type, [&]() {                           \
                                                                             \
    test_triangular_matrix_vector_solve_impl(A_sym_e0, x_e0, x_e0,           \
                         std::experimental::linalg::lower_triangle,          \
                         std::experimental::linalg::implicit_unit_diagonal); \
    test_triangular_matrix_vector_solve_impl(A_sym_e0, x_e0, x_e0,           \
                         std::experimental::linalg::upper_triangle,          \
                         std::experimental::linalg::explicit_diagonal);      \
                                                                             \
  });                                                                        \
}

DEFINE_TESTS(double)
DEFINE_TESTS(float)
DEFINE_TESTS(complex_double)
