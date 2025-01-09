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

template<class A_t, class B_t, class E_t, class C_t, class Triangle>
void updating_hermitian_matrix_left_product_gold_solution(A_t A, Triangle /* t */, B_t B, E_t E, C_t C)
{
  using size_type = typename std::experimental::extents<>::size_type;
  using c_element_type = typename C_t::value_type;
  using std::experimental::linalg::impl::conj_if_needed;
  constexpr bool lower = std::is_same_v<Triangle, std::experimental::linalg::lower_triangle_t>;
  for (size_type i = 0; i < C.extent(0); ++i) {
    for (size_type j = 0; j < C.extent(1); ++j) {
      C(i, j) = E(i, j);
      for (size_type k = 0; k < A.extent(1); ++k) {
        const bool flip = lower ? i <= k : i >= k;
        const auto aik = flip ? conj_if_needed(A(k, i)) : A(i, k);
        C(i, j) += aik * B(k, j);
      }
    }
  }
}

template<class A_t, class B_t, class E_t, class C_t, class Triangle>
void test_updating_hermitian_matrix_left_product_impl(A_t A, B_t B, E_t E, C_t C, Triangle t)
{
  const auto get_gold = [&](auto C_gold) {
      updating_hermitian_matrix_left_product_gold_solution(A, t, B, E, C_gold);
    };
  const auto compute = [&]() {
      std::experimental::linalg::hermitian_matrix_product(
        KokkosKernelsSTD::kokkos_exec<>(), A, t, B, E, C);
    };
  const auto tol = tolerance<typename C_t::value_type>(1e-20, 1e-10f);
  test_op_CAB(A, B, C, tol, get_gold, compute);
}

template<class A_t, class B_t, class C_t, class Triangle>
void test_overwriting_hermitian_matrix_left_product_impl(A_t A, B_t B, C_t C, Triangle t)
{
  const auto get_gold = [&](auto C_gold) {
      set(C_gold, 0);
      updating_hermitian_matrix_left_product_gold_solution(A, t, B, C_gold, C_gold);
    };
  const auto compute = [&]() {
      std::experimental::linalg::hermitian_matrix_product(
        KokkosKernelsSTD::kokkos_exec<>(), A, t, B, C);
    };
  const auto tol = tolerance<typename C_t::value_type>(1e-20, 1e-10f);
  test_op_CAB(A, B, C, tol, get_gold, compute);
}

} // anonymous namespace

#define DEFINE_TESTS(blas_val_type)                                                     \
TEST_F(blas3_signed_##blas_val_type##_fixture,                                          \
       kokkos_hermitian_matrix_left_product) {                                          \
  using val_t = typename blas3_signed_##blas_val_type##_fixture::value_type;            \
  run_checked_tests<val_t>("kokkos_", "hermitian_matrix_left_product", "",              \
                           #blas_val_type, [&]() {                                      \
                                                                                        \
    test_overwriting_hermitian_matrix_left_product_impl(A_hem_e0, B_e0e2, C_e0e2,       \
                            std::experimental::linalg::lower_triangle);                 \
    test_overwriting_hermitian_matrix_left_product_impl(A_hem_e0, B_e0e2, C_e0e2,       \
                            std::experimental::linalg::upper_triangle);                 \
    test_updating_hermitian_matrix_left_product_impl(A_hem_e0, B_e0e2, E_e0e2, C_e0e2,  \
                            std::experimental::linalg::lower_triangle);                 \
    test_updating_hermitian_matrix_left_product_impl(A_hem_e0, B_e0e2, E_e0e2, C_e0e2,  \
                            std::experimental::linalg::upper_triangle);                 \
                                                                                        \
  });                                                                                   \
}

DEFINE_TESTS(double)
DEFINE_TESTS(float)
DEFINE_TESTS(complex_double)