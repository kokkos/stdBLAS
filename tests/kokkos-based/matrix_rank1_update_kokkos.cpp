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

template<class x_t, class y_t, class A_t>
void matrix_rank_1_update_gold_solution(const x_t &x, const y_t &y, A_t &A)
{
  using size_type = std::experimental::extents<>::size_type;
  for (size_type i = 0; i < A.extent(0); ++i) {
    for (size_type j = 0; j < A.extent(1); ++j) {
      A(i, j) += x(i) * y(j);
    }
  }
}

template<class x_t, class y_t, class A_t>
void test_kokkos_matrix_rank1_update_impl(const x_t &x, const y_t &y, A_t &A)
{
  const auto get_gold = [&](auto A_gold) {
      matrix_rank_1_update_gold_solution(x, y, A_gold);
    };
  const auto compute = [&]() {
      std::experimental::linalg::matrix_rank_1_update(
          KokkosKernelsSTD::kokkos_exec<>(), x, y, A);
    };
  const auto tol = tolerance<typename x_t::value_type>(1e-20, 1e-10f);
  test_op_Axy(x, y, A, tol, get_gold, compute);
}

template<class x_t, class y_t, class A_t>
void test_kokkos_matrix_rank1_update_conj_impl(const x_t &x, const y_t &y, A_t &A)
{
  const auto get_gold = [&](auto A_gold) {
      matrix_rank_1_update_gold_solution(x,
        std::experimental::linalg::conjugated(y), A_gold);
    };
  const auto compute = [&]() {
      std::experimental::linalg::matrix_rank_1_update_c(
          KokkosKernelsSTD::kokkos_exec<>(), x, y, A);
    };
  const auto tol = tolerance<typename x_t::value_type>(1e-20, 1e-10f);
  test_op_Axy(x, y, A, tol, get_gold, compute);
}

} // anonymous namespace

#define DEFINE_TESTS(blas_val_type)                                          \
TEST_F(blas2_signed_##blas_val_type##_fixture, kokkos_matrix_rank1_update) { \
  using val_t = typename blas2_signed_##blas_val_type##_fixture::value_type; \
  run_checked_tests<val_t>("kokkos_", "matrix_rank1_update", "",             \
                           #blas_val_type, [&]() {                           \
                                                                             \
   test_kokkos_matrix_rank1_update_impl(x_e0, x_e1, A_e0e1);                 \
                                                                             \
  });                                                                        \
}                                                                            \
TEST_F(blas2_signed_##blas_val_type##_fixture,                               \
       kokkos_matrix_rank1_update_conjugated) {                              \
  using val_t = typename blas2_signed_##blas_val_type##_fixture::value_type; \
  run_checked_tests<val_t>("kokkos_", "matrix_rank1_update", "_conjugated",  \
                           #blas_val_type, [&]() {                           \
                                                                             \
   test_kokkos_matrix_rank1_update_conj_impl(x_e0, x_e1, A_e0e1);            \
                                                                             \
  });                                                                        \
}

DEFINE_TESTS(double)
DEFINE_TESTS(float)
DEFINE_TESTS(complex_double)
