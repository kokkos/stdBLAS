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

template<class x_t, class y_t, class A_t, class Triangle>
void hermitian_matrix_rank_2_update_gold_solution(const x_t &x, const y_t &y, A_t &A, Triangle /* t */)
{
  using std::experimental::linalg::impl::conj_if_needed;
  using size_type = std::experimental::extents<>::size_type;
  constexpr bool low = std::is_same_v<Triangle, std::experimental::linalg::lower_triangle_t>;
  for (size_type j = 0; j < A.extent(1); ++j) {
    const size_type i1 = low ? A.extent(0) : j + 1;
    for (size_type i = low ? j : 0; i < i1; ++i) {
      A(i,j) += x(i) * conj_if_needed(y(j)) + y(i) * conj_if_needed(x(j));
    }
  }
}

template<class x_t, class y_t, class A_t, class Triangle, class Scalar = typename x_t::element_type>
void test_kokkos_hermitian_matrix_rank2_update_impl(const x_t &x, const y_t &y, A_t &A, Triangle t)
{
  const auto get_gold = [&](auto A_gold) {
      hermitian_matrix_rank_2_update_gold_solution(x, y, A_gold, t);
    };
  const auto compute = [&]() {
      std::experimental::linalg::hermitian_matrix_rank_2_update(
        KokkosKernelsSTD::kokkos_exec<>(), x, y, A, t);
    };
  const auto tol = tolerance<typename x_t::value_type>(1e-20, 1e-10f);
  test_op_Axy(x, y, A, tol, get_gold, compute);
}

} // anonymous namespace

#define DEFINE_TESTS(blas_val_type)                                          \
TEST_F(blas2_signed_##blas_val_type##_fixture,                               \
       kokkos_hermitian_matrix_rank2_update) {                               \
  using val_t = typename blas2_signed_##blas_val_type##_fixture::value_type; \
  run_checked_tests<val_t>("kokkos_", "hermitian_matrix_rank2_update", "",   \
                           #blas_val_type, [&]() {                           \
                                                                             \
    test_kokkos_hermitian_matrix_rank2_update_impl(x_e0, y_e0, A_sym_e0,     \
                            std::experimental::linalg::lower_triangle);      \
    test_kokkos_hermitian_matrix_rank2_update_impl(x_e0, y_e0, A_sym_e0,     \
                            std::experimental::linalg::upper_triangle);      \
                                                                             \
  });                                                                        \
}

DEFINE_TESTS(double)
DEFINE_TESTS(float)
DEFINE_TESTS(complex_double)
