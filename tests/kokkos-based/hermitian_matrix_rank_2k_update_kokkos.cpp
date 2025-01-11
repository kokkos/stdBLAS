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

template<class A_t, class B_t, class C_t, class Triangle>
void hermitian_matrix_rank_2k_update_gold_solution(const A_t &A, const B_t &B, C_t &C, Triangle /* t */)
{
  using std::experimental::linalg::impl::conj_if_needed;
  using size_type = std::experimental::extents<>::size_type;
  constexpr bool low = std::is_same_v<Triangle, std::experimental::linalg::lower_triangle_t>;
  const auto size = A.extent(1);
  for (size_type j = 0; j < size; ++j) {
    const size_type i1 = low ? size : j + 1;
    for (size_type i = low ? j : 0; i < i1; ++i) {
      for (size_type k = 0; k < size; ++k) {
        C(i, j) += A(i, k) * conj_if_needed(B(j, k)) + B(i, k) * conj_if_needed(A(j, k));
      }
    }
  }
}

template<class A_t, class B_t, class C_t, class Triangle>
void test_kokkos_hermitian_matrix_rank2k_update_impl(const A_t &A, const B_t &B, C_t &C, Triangle t)
{
  const auto get_gold = [&](auto C_gold) {
      hermitian_matrix_rank_2k_update_gold_solution(A, B, C_gold, t);
    };
  const auto compute = [&]() {
      std::experimental::linalg::hermitian_matrix_rank_2k_update(
        KokkosKernelsSTD::kokkos_exec<>(), A, B, C, t);
    };
  const auto tol = tolerance<typename C_t::value_type>(1e-20, 1e-10f);
  test_op_CAB(A, B, C, tol, get_gold, compute);
}

} // anonymous namespace

#define DEFINE_TESTS(blas_val_type)                                          \
TEST_F(blas2_signed_##blas_val_type##_fixture,                               \
       kokkos_hermitian_matrix_rank2k_update) {                              \
  using val_t = typename blas2_signed_##blas_val_type##_fixture::value_type; \
  run_checked_tests<val_t>("kokkos_", "hermitian_matrix_rank2k_update", "",  \
                           #blas_val_type, [&]() {                           \
                                                                             \
    test_kokkos_hermitian_matrix_rank2k_update_impl(A_sym_e0, A_sym_e0, A_hem_e0,    \
                            std::experimental::linalg::lower_triangle);      \
    test_kokkos_hermitian_matrix_rank2k_update_impl(A_sym_e0, A_sym_e0, A_hem_e0,    \
                            std::experimental::linalg::upper_triangle);      \
                                                                             \
  });                                                                        \
}

DEFINE_TESTS(double)
DEFINE_TESTS(float)
DEFINE_TESTS(complex_double)
