#include "gtest/gtest.h"

#include <experimental/linalg>
#include <experimental/mdspan>
#include <complex>
#include <iostream>
#include <vector>

namespace {
  using std::experimental::mdspan;
  using std::dynamic_extent;
  using std::experimental::extents;
  using std::experimental::layout_left;
  using std::experimental::linalg::explicit_diagonal;
  using std::experimental::linalg::implicit_unit_diagonal;
  using std::experimental::linalg::lower_triangle;
  using std::experimental::linalg::matrix_product;
  using std::experimental::linalg::transposed;
  using std::experimental::linalg::upper_triangle;
  using std::complex;
  using std::cout;
  using std::endl;
  using namespace std::complex_literals;

  #define EXPECT_COMPLEX_NEAR(a, b, tol)  \
    EXPECT_NEAR(a.real(), b.real(), tol); \
    EXPECT_NEAR(a.imag(), b.imag(), tol)

  TEST(BLAS3_hemm, left_lower_tri)
  {
    /* C = A * B, where A is hermitian mxm */
    using extents_t = extents<std::size_t, dynamic_extent, dynamic_extent>;
    using cmatrix_t = mdspan<complex<double>, extents_t, layout_left>;
    using dmatrix_t = mdspan<double, extents_t, layout_left>;
    constexpr double snan = std::numeric_limits<double>::signaling_NaN();

    int m = 3, n = 2;
    std::vector<complex<double>> A_mem(m*m, snan);
    std::vector<double> B_mem(m*n);
    std::vector<complex<double>> C_mem(m*n, snan);
    std::vector<complex<double>> gs_mem(m*n);

    cmatrix_t A(A_mem.data(), m, m);
    dmatrix_t B(B_mem.data(), m, n);
    cmatrix_t C(C_mem.data(), m, n);
    cmatrix_t gs(gs_mem.data(), m, n);

    // Fill A
    A[0,0] = -4.0;
    A[1,0] = 4.0 + 4.4i;
    A[1,1] = 3.5;
    A[2,0] = 4.4 + 2.2i;
    A[2,1] = -2.8 - 4.0i;
    A[2,2] = -1.2;
    
    // Fill B
    B[0,0] = 1.3;
    B[0,1] = 2.5;
    B[1,0] = -4.6;
    B[1,1] = -3.7;
    B[2,0] = 3.1;
    B[2,1] = -1.5;

    // Fill GS
    gs[0,0] = -9.96 + 13.42i;
    gs[0,1] = -31.4 + 19.58i;
    gs[1,0] = -19.58 + 18.12i;
    gs[1,1] = 1.25 + 5.0i;
    gs[2,0] = 14.88 + 21.26i;
    gs[2,1] = 23.16 + 20.3i;

    hermitian_matrix_left_product(A, lower_triangle, B, C);

    // TODO: Choose a more reasonable value
    constexpr double TOL = 1e-9;
    for (ptrdiff_t j = 0; j < n; ++j) {
      for (ptrdiff_t i = 0; i < m; ++i) {
        EXPECT_COMPLEX_NEAR((gs[i,j]), (C[i,j]), TOL) 
          << "Matrices differ at index (" 
          << i << "," << j << ")\n";
      }
    }
  }

  TEST(BLAS3_hemm, left_upper_tri)
  {
    /* C = A * B, where A is hermitian mxm */
    using extents_t = extents<std::size_t, dynamic_extent, dynamic_extent>;
    using cmatrix_t = mdspan<complex<double>, extents_t, layout_left>;
    using dmatrix_t = mdspan<double, extents_t, layout_left>;
    constexpr double snan = std::numeric_limits<double>::signaling_NaN();

    int m = 3, n = 2;
    std::vector<complex<double>> A_mem(m*m, snan);
    std::vector<double> B_mem(m*n);
    std::vector<complex<double>> C_mem(m*n, snan);
    std::vector<complex<double>> gs_mem(m*n);

    cmatrix_t A(A_mem.data(), m, m);
    dmatrix_t B(B_mem.data(), m, n);
    cmatrix_t C(C_mem.data(), m, n);
    cmatrix_t gs(gs_mem.data(), m, n);

    // Fill A
    A[0,0] = -4.0;
    A[0,1] = 4.0 - 4.4i;
    A[1,1] = 3.5;
    A[0,2] = 4.4 - 2.2i;
    A[1,2] = -2.8 + 4.0i;
    A[2,2] = -1.2;
    
    // Fill B
    B[0,0] = 1.3;
    B[0,1] = 2.5;
    B[1,0] = -4.6;
    B[1,1] = -3.7;
    B[2,0] = 3.1;
    B[2,1] = -1.5;

    // Fill GS
    gs[0,0] = -9.96 + 13.42i;
    gs[0,1] = -31.4 + 19.58i;
    gs[1,0] = -19.58 + 18.12i;
    gs[1,1] = 1.25 + 5.0i;
    gs[2,0] = 14.88 + 21.26i;
    gs[2,1] = 23.16 + 20.3i;

    hermitian_matrix_left_product(A, upper_triangle, B, C);

    // TODO: Choose a more reasonable value
    constexpr double TOL = 1e-9;
    for (ptrdiff_t j = 0; j < n; ++j) {
      for (ptrdiff_t i = 0; i < m; ++i) {
        EXPECT_COMPLEX_NEAR((gs[i,j]), (C[i,j]), TOL) 
          << "Matrices differ at index (" 
          << i << "," << j << ")\n";
      }
    }
  }

  TEST(BLAS3_hemm, right_lower_tri)
  {
    /* C = B * A, where A is hermitian mxm */
    using extents_t = extents<std::size_t, dynamic_extent, dynamic_extent>;
    using cmatrix_t = mdspan<complex<double>, extents_t, layout_left>;
    using dmatrix_t = mdspan<double, extents_t, layout_left>;
    constexpr double snan = std::numeric_limits<double>::signaling_NaN();

    int m = 3, n = 2;
    std::vector<complex<double>> A_mem(m*m, snan);
    std::vector<double> B_mem(m*n);
    std::vector<complex<double>> C_mem(m*n, snan);
    std::vector<complex<double>> gs_mem(m*n);

    cmatrix_t A(A_mem.data(), m, m);
    dmatrix_t B(B_mem.data(), n, m);
    cmatrix_t C(C_mem.data(), n, m);
    cmatrix_t gs(gs_mem.data(), n, m);

    // Fill A
    A[0,0] = -4.0;
    A[1,0] = 4.0 + 4.4i;
    A[1,1] = 3.5;
    A[2,0] = 4.4 + 2.2i;
    A[2,1] = -2.8 - 4.0i;
    A[2,2] = -1.2;
    
    // Fill B
    B[0,0] = 1.3;
    B[1,0] = 2.5;
    B[0,1] = -4.6;
    B[1,1] = -3.7;
    B[0,2] = 3.1;
    B[1,2] = -1.5;

    // Fill GS
    gs[0,0] = -9.96 - 13.42i;
    gs[1,0] = -31.4 - 19.58i;
    gs[0,1] = -19.58 - 18.12i;
    gs[1,1] = 1.25 - 5.0i;
    gs[0,2] = 14.88 - 21.26i;
    gs[1,2] = 23.16 - 20.3i;

    hermitian_matrix_right_product(A, lower_triangle, B, C);

    // TODO: Choose a more reasonable value
    constexpr double TOL = 1e-9;
    for (ptrdiff_t j = 0; j < m; ++j) {
      for (ptrdiff_t i = 0; i < n; ++i) {
        EXPECT_COMPLEX_NEAR((gs[i,j]), (C[i,j]), TOL) 
          << "Matrices differ at index (" 
          << i << "," << j << ")\n";
      }
    }
  }

  TEST(BLAS3_hemm, right_upper_tri)
  {
    /* C = B * A, where A is hermitian mxm */
    using extents_t = extents<std::size_t, dynamic_extent, dynamic_extent>;
    using cmatrix_t = mdspan<complex<double>, extents_t, layout_left>;
    using dmatrix_t = mdspan<double, extents_t, layout_left>;
    constexpr double snan = std::numeric_limits<double>::signaling_NaN();

    int m = 3, n = 2;
    std::vector<complex<double>> A_mem(m*m, snan);
    std::vector<double> B_mem(m*n);
    std::vector<complex<double>> C_mem(m*n, snan);
    std::vector<complex<double>> gs_mem(m*n);

    cmatrix_t A(A_mem.data(), m, m);
    dmatrix_t B(B_mem.data(), n, m);
    cmatrix_t C(C_mem.data(), n, m);
    cmatrix_t gs(gs_mem.data(), n, m);

    // Fill A
    A[0,0] = -4.0;
    A[0,1] = 4.0 - 4.4i;
    A[1,1] = 3.5;
    A[0,2] = 4.4 - 2.2i;
    A[1,2] = -2.8 + 4.0i;
    A[2,2] = -1.2;
    
    // Fill B
    B[0,0] = 1.3;
    B[1,0] = 2.5;
    B[0,1] = -4.6;
    B[1,1] = -3.7;
    B[0,2] = 3.1;
    B[1,2] = -1.5;

    // Fill GS
    gs[0,0] = -9.96 - 13.42i;
    gs[1,0] = -31.4 - 19.58i;
    gs[0,1] = -19.58 - 18.12i;
    gs[1,1] = 1.25 - 5.0i;
    gs[0,2] = 14.88 - 21.26i;
    gs[1,2] = 23.16 - 20.3i;

    hermitian_matrix_right_product(A, upper_triangle, B, C);

    // TODO: Choose a more reasonable value
    constexpr double TOL = 1e-9;
    for (ptrdiff_t j = 0; j < m; ++j) {
      for (ptrdiff_t i = 0; i < n; ++i) {
        EXPECT_COMPLEX_NEAR((gs[i,j]), (C[i,j]), TOL) 
          << "Matrices differ at index (" 
          << i << "," << j << ")\n";
      }
    }
  }
} // end anonymous namespace
