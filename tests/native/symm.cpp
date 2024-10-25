#include "./gtest_fixtures.hpp"
#include <iostream>

namespace {
  using LinearAlgebra::explicit_diagonal;
  using LinearAlgebra::implicit_unit_diagonal;
  using LinearAlgebra::lower_triangle;
  using LinearAlgebra::matrix_product;
  using LinearAlgebra::transposed;
  using LinearAlgebra::upper_triangle;
  using std::complex;
  using std::cout;
  using std::endl;
  using namespace std::complex_literals;

  #define EXPECT_COMPLEX_NEAR(a, b, tol)  \
    EXPECT_NEAR(a.real(), b.real(), tol); \
    EXPECT_NEAR(a.imag(), b.imag(), tol)

  template<bool isLeft, typename Triangle, bool isUpdating>
  void test_symmetric_matrix_product()
  {
    /* C = A * B, where A is symmetric mxm */
    using extents_t = extents<std::size_t, dynamic_extent, dynamic_extent>;
    using cmatrix_t = mdspan<complex<double>, extents_t, layout_left>;
    using dmatrix_t = mdspan<double, extents_t, layout_left>;
    constexpr double snan = std::numeric_limits<double>::signaling_NaN();

    int m = 3, n = 2;
    std::vector<complex<double>> A_mem(m*m, snan);
    std::vector<double> B_mem(m*n);
    std::vector<complex<double>> E_mem(m*n);
    std::vector<complex<double>> C_mem(m*n, snan);
    std::vector<complex<double>> gs_mem(m*n);

    cmatrix_t A(A_mem.data(), m, m);
    dmatrix_t B(B_mem.data(), (isLeft)?m:n, (isLeft)?n:m);
    cmatrix_t E(E_mem.data(), (isLeft)?m:n, (isLeft)?n:m);
    cmatrix_t C(C_mem.data(), (isLeft)?m:n, (isLeft)?n:m);
    cmatrix_t gs(gs_mem.data(), (isLeft)?m:n, (isLeft)?n:m);

    // Fill A
    if(std::is_same_v<Triangle, decltype(lower_triangle)>) {
      A(0,0) = -4.0 + 0.9i;
      A(1,0) = 4.0 + 4.4i;
      A(1,1) = 3.5 - 4.2i;
      A(2,0) = 4.4 + 2.2i;
      A(2,1) = -2.8 - 4.0i;
      A(2,2) = -1.2 + 1.7i;
    }
    else {
      A(0,0) = -4.0 + 0.9i;
      A(0,1) = 4.0 + 4.4i;
      A(1,1) = 3.5 - 4.2i;
      A(0,2) = 4.4 + 2.2i;
      A(1,2) = -2.8 - 4.0i;
      A(2,2) = -1.2 + 1.7i;
    }

    // Fill B
    if(isLeft) {
      B(0,0) = 1.3;
      B(0,1) = 2.5;
      B(1,0) = -4.6;
      B(1,1) = -3.7;
      B(2,0) = 3.1;
      B(2,1) = -1.5;
    }
    else {
      B(0,0) = 1.3;
      B(1,0) = 2.5;
      B(0,1) = -4.6;
      B(1,1) = -3.7;
      B(0,2) = 3.1;
      B(1,2) = -1.5;
    }

    if(isUpdating) {
      for (ptrdiff_t j = 0; j < (isLeft?n:m); ++j) {
        for (ptrdiff_t i = 0; i < (isLeft?m:n); ++i) {
          E(i,j) = B(i,j);
        }
      }
    }

    // Fill GS
    if(isLeft) {
      gs(0,0) = E(0,0) - 9.96 - 12.25i;
      gs(0,1) = E(0,1) - 31.4 - 17.33i;
      gs(1,0) = E(1,0) - 19.58 + 12.64i;
      gs(1,1) = E(1,1) + 1.25 + 32.54i;
      gs(2,0) = E(2,0) + 14.88 + 26.53i;
      gs(2,1) = E(2,1) + 23.16 + 17.75i;
    }
    else {
      gs(0,0) = E(0,0) - 9.96 - 12.25i;
      gs(1,0) = E(1,0) - 31.4 - 17.33i;
      gs(0,1) = E(0,1) - 19.58 + 12.64i;
      gs(1,1) = E(1,1) + 1.25 + 32.54i;
      gs(0,2) = E(0,2) + 14.88 + 26.53i;
      gs(1,2) = E(1,2) + 23.16 + 17.75i;
    }

    if(isLeft) {
      if(isUpdating) {
        symmetric_matrix_product(A, Triangle{}, B, E, C);
      }
      else {
        symmetric_matrix_product(A, Triangle{}, B, C);
      }
    }
    else {
      if(isUpdating) {
        symmetric_matrix_product(B, A, Triangle{}, E, C);
      }
      else {
        symmetric_matrix_product(B, A, Triangle{}, C);
      }
    }

    // TODO: Choose a more reasonable value
    constexpr double TOL = 1e-9;
    for (ptrdiff_t j = 0; j < (isLeft?n:m); ++j) {
      for (ptrdiff_t i = 0; i < (isLeft?m:n); ++i) {
        EXPECT_COMPLEX_NEAR(gs(i,j), C(i,j), TOL)
          << "Matrices differ at index ("
          << i << "," << j << ")\n";
      }
    }
  }

  TEST(BLAS3_symm, left_lower_tri)
  {
    test_symmetric_matrix_product<true, decltype(lower_triangle), false>();
  }

  TEST(BLAS3_symm, left_upper_tri)
  {
    test_symmetric_matrix_product<true, decltype(upper_triangle), false>();
  }

  TEST(BLAS3_symm, right_lower_tri)
  {
    test_symmetric_matrix_product<false, decltype(lower_triangle), false>();
  }

  TEST(BLAS3_symm, right_upper_tri)
  {
    test_symmetric_matrix_product<false, decltype(upper_triangle), false>();
  }

  TEST(BLAS3_symm, left_lower_tri_upd)
  {
    test_symmetric_matrix_product<true, decltype(lower_triangle), true>();
  }

  TEST(BLAS3_symm, left_upper_tri_upd)
  {
    test_symmetric_matrix_product<true, decltype(upper_triangle), true>();
  }

  TEST(BLAS3_symm, right_lower_tri_upd)
  {
    test_symmetric_matrix_product<false, decltype(lower_triangle), true>();
  }

  TEST(BLAS3_symm, right_upper_tri_upd)
  {
    test_symmetric_matrix_product<false, decltype(upper_triangle), true>();
  }
} // end anonymous namespace
