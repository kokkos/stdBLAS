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
