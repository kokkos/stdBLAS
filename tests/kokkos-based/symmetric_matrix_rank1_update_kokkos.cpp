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

template<class x_t, class A_t, class Triangle>
void symmetric_matrix_rank_1_update_gold_solution(const x_t &x, A_t &A, Triangle /* t */)
{
  using size_type = std::experimental::extents<>::size_type;
  constexpr bool low = std::is_same_v<Triangle, std::experimental::linalg::lower_triangle_t>;
  for (size_type j = 0; j < A.extent(1); ++j) {
    const size_type i1 = low ? A.extent(0) : j + 1;
    for (size_type i = low ? j : 0; i < i1; ++i) {
      A(i,j) += x(i) * x(j);
    }
  }
}

template<class x_t, class A_t, class Triangle, class Scalar = typename x_t::element_type>
void test_kokkos_symmetric_matrix_rank1_update_impl(const x_t &x, A_t &A, Triangle t)
{
  const auto get_gold = [&](auto A_gold) {
      symmetric_matrix_rank_1_update_gold_solution(x, A_gold, t);
    };
  const auto compute = [&]() {
      std::experimental::linalg::symmetric_matrix_rank_1_update(
        KokkosKernelsSTD::kokkos_exec<>(), x, A, t);
    };
  const auto tol = tolerance<typename x_t::value_type>(1e-9, 1e-2f);
  test_op_Ax(x, A, tol, get_gold, compute);
}

} // anonymous namespace

#define DEFINE_TESTS(blas_val_type)                                          \
TEST_F(blas2_signed_##blas_val_type##_fixture,                               \
       kokkos_symmetric_matrix_rank1_update) {                               \
  using val_t = typename blas2_signed_##blas_val_type##_fixture::value_type; \
  run_checked_tests<val_t>("kokkos_", "symmetric_matrix_rank1_update", "",   \
                           #blas_val_type, [&]() {                           \
                                                                             \
    test_kokkos_symmetric_matrix_rank1_update_impl(x_e0, A_sym_e0,           \
                            std::experimental::linalg::lower_triangle);      \
    test_kokkos_symmetric_matrix_rank1_update_impl(x_e0, A_sym_e0,           \
                            std::experimental::linalg::upper_triangle);      \
                                                                             \
  });                                                                        \
}

DEFINE_TESTS(double)
DEFINE_TESTS(float)
DEFINE_TESTS(complex_double)
