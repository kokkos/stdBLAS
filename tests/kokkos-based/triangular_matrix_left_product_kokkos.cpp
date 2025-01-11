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

template<class A_t, class B_t, class E_t, class C_t, class Triangle, class Diagonal>
void updating_triangular_matrix_left_product_gold_solution(A_t A,
        Triangle /* t */, Diagonal /* d */, B_t B, E_t E, C_t C)
{
  using size_type = typename extents<>::size_type;
  using c_element_type = typename C_t::value_type;
  constexpr bool explicit_diag = std::is_same_v<Diagonal,
      std::experimental::linalg::explicit_diagonal_t>;
  constexpr bool lower = std::is_same_v<Triangle,
      std::experimental::linalg::lower_triangle_t>;
  const auto C_ext0 = C.extent(0); // == A_ext0 == A_ext1
  const auto C_ext1 = C.extent(1);

  // Note: This routine can be called in-place (B=C)
  // because (i,j) indexing respects C updating order
  // and parallelism is restricted accordingly.
  for (size_type ii = 0; ii < C_ext0; ++ii) {
    const size_type i = lower ? C_ext0 - 1 - ii : ii;
    for (size_type j = 0; j < C_ext1; ++j) {
      c_element_type t = E(i, j);
      const size_type k0 = lower ? 0 : (explicit_diag ? i : i + 1);
      const size_type k1 = lower ? (explicit_diag ? i + 1 : i) : C_ext0;
      for (size_type k = k0; k < k1; ++k) {
        t += A(i, k) * B(k, j);
      }
      if constexpr (!explicit_diag) {
        t += /* 1 times */ B(i, j);
      }
      C(i, j) = t;
    }
  }
}

template<class A_t, class B_t, class C_t, class Triangle, class Diagonal>
void test_overwriting_triangular_matrix_left_product_impl(A_t A, B_t B, C_t C, Triangle t, Diagonal d)
{
  const auto get_gold = [&](auto C_gold) {
      set(C_gold, 0);
      updating_triangular_matrix_left_product_gold_solution(A, t, d, B, C_gold, C_gold);
    };
  const auto compute = [&]() {
      std::experimental::linalg::triangular_matrix_product(
        KokkosKernelsSTD::kokkos_exec<>(), A, t, d, B, C);
    };
  const auto tol = tolerance<typename C_t::value_type>(1e-20, 1e-10f);
  test_op_CAB(A, B, C, tol, get_gold, compute);
}

template<class A_t, class C_t, class Triangle, class Diagonal>
void test_overwriting_triangular_matrix_left_product_impl(A_t A, C_t C, Triangle t, Diagonal d)
{
  const auto get_gold = [&](auto C_gold) {
      set(C_gold, 0);
      updating_triangular_matrix_left_product_gold_solution(A, t, d, C, C_gold, C_gold);
    };
  const auto compute = [&]() {
      std::experimental::linalg::triangular_matrix_product(
        KokkosKernelsSTD::kokkos_exec<>(), A, t, d, C);
    };
  const auto tol = tolerance<typename C_t::value_type>(1e-20, 1e-10f);
  test_op_CA(A, C, tol, get_gold, compute);
}

} // anonymous namespace

#define DEFINE_TESTS(blas_val_type)                                                     \
TEST_F(blas3_signed_##blas_val_type##_fixture,                                          \
       kokkos_triangular_matrix_left_product) {                                         \
  using val_t = typename blas3_signed_##blas_val_type##_fixture::value_type;            \
  run_checked_tests<val_t>("kokkos_", "triangular_matrix_left_product", "",             \
                           #blas_val_type, [&]() {                                      \
    /* overwriting, not-in-place */                                                     \
    test_overwriting_triangular_matrix_left_product_impl(A_sym_e0, B_e0e2, C_e0e2,      \
                            std::experimental::linalg::lower_triangle,                  \
                            std::experimental::linalg::implicit_unit_diagonal);         \
    test_overwriting_triangular_matrix_left_product_impl(A_sym_e0, B_e0e2, C_e0e2,      \
                            std::experimental::linalg::upper_triangle,                  \
                            std::experimental::linalg::explicit_diagonal);              \
    /* overwriting, in-place */                                                         \
    test_overwriting_triangular_matrix_left_product_impl(A_sym_e0, C_e0e2,              \
                            std::experimental::linalg::lower_triangle,                  \
                            std::experimental::linalg::implicit_unit_diagonal);         \
    test_overwriting_triangular_matrix_left_product_impl(A_sym_e0, C_e0e2,              \
                            std::experimental::linalg::upper_triangle,                  \
                            std::experimental::linalg::explicit_diagonal);              \
                                                                                        \
  });                                                                                   \
}

DEFINE_TESTS(double)
DEFINE_TESTS(float)
DEFINE_TESTS(complex_double)