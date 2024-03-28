#include "./gtest_fixtures.hpp"
#include <iostream>

#if (! defined(__GNUC__)) || (__GNUC__ > 9)
#  define MDSPAN_EXAMPLES_USE_EXECUTION_POLICIES 1
#endif
#ifdef MDSPAN_EXAMPLES_USE_EXECUTION_POLICIES
#  include <execution>
#endif

namespace {

using LinearAlgebra::matrix_vector_product;
using LinearAlgebra::scaled;

TEST(gemv, no_ambiguity)
{
  int N = 40, M = 20;
  {
    // Create Data
    std::vector<double> A_vec(N*M);
    std::vector<double> x_vec(M);
    std::vector<double> y_vec(N);

    mdspan<double, extents<std::size_t, dynamic_extent,dynamic_extent>> A(A_vec.data(), N, M);
    mdspan<double, extents<std::size_t, dynamic_extent>> x(x_vec.data(), M);
    mdspan<double, extents<std::size_t, dynamic_extent>> y(y_vec.data(), N);
    for (int i = 0; i < A.extent(0); ++i) {
      for (int j = 0; j < A.extent(1); ++j) {
        A(i,j) = 100.0 * i + j;
      }
    }
    for(int i = 0; i < x.extent(0); ++i) {
      x(i) = 1.0 * i;
    }
    for(int i = 0; i < y.extent(0); ++i) {
      y(i) = -1.0 * i;
    }

    matrix_vector_product(A, x, y);
    // The following is an ambiguous call unless the implementation
    // correctly constraints ExecutionPolicy (the spec would imply
    // std::is_execution_policy_v, though implementations might define
    // their own custom "execution policies" that cannot satisfy this).
    matrix_vector_product(
       scaled(2.0, A), x,
       scaled(0.5, y), y);

#ifdef MDSPAN_EXAMPLES_USE_EXECUTION_POLICIES
    matrix_vector_product(std::execution::par,
       scaled(2.0, A), x,
       scaled(0.5, y), y);
#endif
  }
}

}
