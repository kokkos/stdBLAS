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
void updating_triangular_matrix_right_product_gold_solution(A_t A,
        Triangle /* t */, Diagonal /* d */, B_t B, E_t E, C_t C)
{
  using size_type = typename extents<>::size_type;
  using c_element_type = typename C_t::value_type;
  constexpr bool explicit_diag = std::is_same_v<Diagonal,
      std::experimental::linalg::explicit_diagonal_t>;
  constexpr bool lower = std::is_same_v<Triangle,
      std::experimental::linalg::lower_triangle_t>;
  const auto C_ext0 = C.extent(0);
  const auto C_ext1 = C.extent(1); // == A_ext0 == A_ext1

  // Note: This routine can be called in-place (B=C)
  // because (i,j) indexing respects C updating order
  // and parallelism is restricted accordingly.
  for (size_type jj = 0; jj < C_ext1; ++jj) {
    const size_type j = lower ? jj : C_ext1 - 1 - jj;
    for (size_type i = 0; i < C_ext0; ++i) {
      c_element_type t = E(i, j);
      // Note: lower triangle of A(k, j) means k <= j
      const auto k0 = lower ? (explicit_diag ? j : j + 1) : 0;
      const auto k1 = lower ? C_ext1 : (explicit_diag ? j + 1 : j);
      for (size_type k = k0; k < k1; ++k) {
        t += B(i, k) * A(k, j);
      }
      if constexpr (!explicit_diag) {
        t += B(i, j) /* times 1 */;
      }
      C(i, j) = t;
    }
  }
}

template<class A_t, class B_t, class C_t, class Triangle, class Diagonal>
void test_overwriting_triangular_matrix_right_product_impl(A_t A, B_t B, C_t C, Triangle t, Diagonal d)
{
  const auto get_gold = [&](auto C_gold) {
      set(C_gold, 0);
      updating_triangular_matrix_right_product_gold_solution(A, t, d, B, C_gold, C_gold);
    };
  const auto compute = [&]() {
      std::experimental::linalg::triangular_matrix_product(
        KokkosKernelsSTD::kokkos_exec<>(), B, A, t, d, C);
    };
  const auto tol = tolerance<typename C_t::value_type>(1e-20, 1e-10f);
  test_op_CAB(A, B, C, tol, get_gold, compute);
}

// in-place
template<class A_t, class C_t, class Triangle, class Diagonal>
void test_overwriting_triangular_matrix_right_product_impl(A_t A, C_t C, Triangle t, Diagonal d)
{
  const auto get_gold = [&](auto C_gold) {
      set(C_gold, 0);
      updating_triangular_matrix_right_product_gold_solution(A, t, d, C, C_gold, C_gold);
    };
  const auto compute = [&]() {
      std::experimental::linalg::triangular_matrix_product(
        KokkosKernelsSTD::kokkos_exec<>(), A, t, d, C);
    };
  const auto tol = tolerance<typename C_t::value_type>(1e-20, 1e-10f);
  test_op_CA(A, C, tol, get_gold, compute);
}

} // anonymous namespace

#define DEFINE_TESTS(blas_val_type)                                                 \
TEST_F(blas3_signed_##blas_val_type##_fixture,                                      \
       kokkos_triangular_matrix_right_product) {                                    \
  using val_t = typename blas3_signed_##blas_val_type##_fixture::value_type;        \
  run_checked_tests<val_t>("kokkos_", "triangular_matrix_right_product", "",        \
                           #blas_val_type, [&]() {                                  \
    /* copy to prevent fixture modification (TODO: add to fixture ?) */             \
    auto B_data = create_stdvector_and_copy_rowwise(C_e2e0);                        \
    auto B_e2e0 = make_mdspan(B_data.data(), C_e2e0.extent(0), C_e2e0.extent(1));   \
                                                                                    \
    /* overwriting, not-in-place */                                                 \
    test_overwriting_triangular_matrix_right_product_impl(A_sym_e0, B_e2e0, C_e2e0, \
                            std::experimental::linalg::lower_triangle,              \
                            std::experimental::linalg::implicit_unit_diagonal);     \
    test_overwriting_triangular_matrix_right_product_impl(A_sym_e0, B_e2e0, C_e2e0, \
                            std::experimental::linalg::upper_triangle,              \
                            std::experimental::linalg::explicit_diagonal);          \
    /* overwriting, in-place */                                                     \
    test_overwriting_triangular_matrix_right_product_impl(A_sym_e0, C_e2e0,         \
                            std::experimental::linalg::lower_triangle,              \
                            std::experimental::linalg::implicit_unit_diagonal);     \
    test_overwriting_triangular_matrix_right_product_impl(A_sym_e0, C_e2e0,         \
                            std::experimental::linalg::upper_triangle,              \
                            std::experimental::linalg::explicit_diagonal);          \
                                                                                    \
  });                                                                               \
}

DEFINE_TESTS(double)
DEFINE_TESTS(float)
DEFINE_TESTS(complex_double)