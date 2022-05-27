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
