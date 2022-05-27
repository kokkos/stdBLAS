 /*
//@HEADER
// ************************************************************************
//
//                        Kokkos v. 2.0
//              Copyright (2019) Sandia Corporation
//
// Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
// the U.S. Government retains certain rights in this software.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
//
// 1. Redistributions of source code must retain the above copyright
// notice, this list of conditions and the following disclaimer.
//
// 2. Redistributions in binary form must reproduce the above copyright
// notice, this list of conditions and the following disclaimer in the
// documentation and/or other materials provided with the distribution.
//
// 3. Neither the name of the Corporation nor the names of the
// contributors may be used to endorse or promote products derived from
// this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY SANDIA CORPORATION "AS IS" AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL SANDIA CORPORATION OR THE
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
// LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
// NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
// SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
// Questions? Contact Christian R. Trott (crtrott@sandia.gov)
//
// ************************************************************************
//@HEADER
*/

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
      std::experimental::linalg::triangular_matrix_left_product(
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
      std::experimental::linalg::triangular_matrix_left_product(
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