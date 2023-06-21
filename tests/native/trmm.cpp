#define MDSPAN_USE_PAREN_OPERATOR 1
#include "gtest/gtest.h"

#include <experimental/linalg>
#include <experimental/mdspan>
#include <iostream>
#include <vector>

namespace {
  using std::experimental::linalg::explicit_diagonal;
  using std::experimental::linalg::implicit_unit_diagonal;
  using std::experimental::linalg::lower_triangle;
  using std::experimental::linalg::matrix_product;
  using std::experimental::linalg::transposed;
  using std::experimental::linalg::upper_triangle;
  using MDSPAN_IMPL_STANDARD_NAMESPACE::dextents;
  using MDSPAN_IMPL_STANDARD_NAMESPACE::dynamic_extent;
  using MDSPAN_IMPL_STANDARD_NAMESPACE::extents;
  using MDSPAN_IMPL_STANDARD_NAMESPACE::layout_left;
  using MDSPAN_IMPL_STANDARD_NAMESPACE::mdspan;
  using std::cout;
  using std::endl;

  TEST(BLAS3_trmm, left_lower_tri_explicit_diag)
  {
    /* C = A * B, where A is triangular mxm */
    using extents_t = dextents<std::size_t, 2>;
    using matrix_t = mdspan<double, extents_t, layout_left>;
    constexpr double snan = std::numeric_limits<double>::signaling_NaN();

    int m = 3, n = 2;
    std::vector<double> A_mem(m*m, snan);
    std::vector<double> B_mem(m*n);
    std::vector<double> C_mem(m*n, snan);
    std::vector<double> gs_mem(m*n);

    matrix_t A(A_mem.data(), m, m);
    matrix_t B(B_mem.data(), m, n);
    matrix_t C(C_mem.data(), m, n);
    matrix_t gs(gs_mem.data(), m, n);

    // Fill A
    A(0,0) = 3.5;
    A(1,0) = -2.0;
    A(1,1) = 1.2;
    A(2,0) = -0.1;
    A(2,1) = 4.5;
    A(2,2) = -1.0;

    // Fill B
    B(0,0) = -4.4;
    B(0,1) = 1.8;
    B(1,0) = -1.4;
    B(1,1) = 3.4;
    B(2,0) = 1.8;
    B(2,1) = 1.6;

    // Fill GS
    gs(0,0) = -15.4;
    gs(0,1) = 6.3;
    gs(1,0) = 7.12;
    gs(1,1) = 0.48;
    gs(2,0) = -7.66;
    gs(2,1) = 13.52;

    // Check the non-overwriting version
    triangular_matrix_left_product(A, lower_triangle, explicit_diagonal, B, C);

    for (ptrdiff_t j = 0; j < n; ++j) {
      for (ptrdiff_t i = 0; i < m; ++i) {
        EXPECT_DOUBLE_EQ(gs(i,j), C(i,j))
          << "Matrices differ at index ("
          << i << "," << j << ")\n";
      }
    }

    // Check the overwriting version
    triangular_matrix_left_product(A, lower_triangle, explicit_diagonal, B);

    for (ptrdiff_t j = 0; j < n; ++j) {
      for (ptrdiff_t i = 0; i < m; ++i) {
        EXPECT_DOUBLE_EQ(gs(i,j), B(i,j))
          << "Matrices differ at index ("
          << i << "," << j << ")\n";
      }
    }
  }

TEST(BLAS3_trmm, left_lower_tri_implicit_diag)
  {
    /* C = A * B, where A is triangular mxm */
    using extents_t = dextents<std::size_t, 2>;
    using matrix_t = mdspan<double, extents_t, layout_left>;
    constexpr double snan = std::numeric_limits<double>::signaling_NaN();

    int m = 3, n = 2;
    std::vector<double> A_mem(m*m, snan);
    std::vector<double> B_mem(m*n);
    std::vector<double> C_mem(m*n, snan);
    std::vector<double> gs_mem(m*n);

    matrix_t A(A_mem.data(), m, m);
    matrix_t B(B_mem.data(), m, n);
    matrix_t C(C_mem.data(), m, n);
    matrix_t gs(gs_mem.data(), m, n);

    // Fill A
    A(1,0) = -2.0;
    A(2,0) = -0.1;
    A(2,1) = 4.5;

    // Fill B
    B(0,0) = -4.4;
    B(0,1) = 1.8;
    B(1,0) = -1.4;
    B(1,1) = 3.4;
    B(2,0) = 1.8;
    B(2,1) = 1.6;

    triangular_matrix_left_product(A, lower_triangle, implicit_unit_diagonal, B, C);

    // Fill GS
    gs(0,0) = -4.4;
    gs(0,1) = 1.8;
    gs(1,0) = 7.4;
    gs(1,1) = -0.2;
    gs(2,0) = -4.06;
    gs(2,1) = 16.72;

    for (ptrdiff_t j = 0; j < n; ++j) {
      for (ptrdiff_t i = 0; i < m; ++i) {
        // FIXME: Choose a more reasonable value for the tolerance
        constexpr double tol = 1e-9;
        EXPECT_NEAR(gs(i,j), C(i,j), tol)
          << "Matrices differ at index ("
          << i << "," << j << ")\n";
      }
    }

    // Check the overwriting version
    triangular_matrix_left_product(A, lower_triangle, implicit_unit_diagonal, B);

    for (ptrdiff_t j = 0; j < n; ++j) {
      for (ptrdiff_t i = 0; i < m; ++i) {
        // FIXME: Choose a more reasonable value for the tolerance
        constexpr double tol = 1e-9;
        EXPECT_NEAR(gs(i,j), B(i,j), tol)
          << "Matrices differ at index ("
          << i << "," << j << ")\n";
      }
    }
  }

TEST(BLAS3_trmm, left_upper_tri_explicit_diag)
  {
    /* C = A * B, where A is triangular mxm */
    using extents_t = dextents<std::size_t, 2>;
    using matrix_t = mdspan<double, extents_t, layout_left>;
    constexpr double snan = std::numeric_limits<double>::signaling_NaN();

    int m = 3, n = 2;
    std::vector<double> A_mem(m*m, snan);
    std::vector<double> B_mem(m*n);
    std::vector<double> C_mem(m*n, snan);
    std::vector<double> gs_mem(m*n);

    matrix_t A(A_mem.data(), m, m);
    matrix_t B(B_mem.data(), m, n);
    matrix_t C(C_mem.data(), m, n);
    matrix_t gs(gs_mem.data(), m, n);

    // Fill A
    A(0,0) = 3.5;
    A(0,1) = -2.0;
    A(1,1) = 1.2;
    A(0,2) = -0.1;
    A(1,2) = 4.5;
    A(2,2) = -1.0;

    // Fill B
    B(0,0) = -4.4;
    B(0,1) = 1.8;
    B(1,0) = -1.4;
    B(1,1) = 3.4;
    B(2,0) = 1.8;
    B(2,1) = 1.6;

    // Fill GS
    gs(0,0) = -12.78;
    gs(0,1) = -0.66;
    gs(1,0) = 6.42;
    gs(1,1) = 11.28;
    gs(2,0) = -1.8;
    gs(2,1) = -1.6;

    // Check the non-overwriting version
    triangular_matrix_left_product(A, upper_triangle, explicit_diagonal, B, C);

    for (ptrdiff_t j = 0; j < n; ++j) {
      for (ptrdiff_t i = 0; i < m; ++i) {
        EXPECT_DOUBLE_EQ(gs(i,j), C(i,j))
          << "Matrices differ at index ("
          << i << "," << j << ")\n";
      }
    }

    // Check the overwriting version
    triangular_matrix_left_product(A, upper_triangle, explicit_diagonal, B);

    for (ptrdiff_t j = 0; j < n; ++j) {
      for (ptrdiff_t i = 0; i < m; ++i) {
        EXPECT_DOUBLE_EQ(gs(i,j), B(i,j))
          << "Matrices differ at index ("
          << i << "," << j << ")\n";
      }
    }
  }

TEST(BLAS3_trmm, left_upper_tri_implicit_diag)
  {
    /* C = A * B, where A is triangular mxm */
    using extents_t = dextents<std::size_t, 2>;
    using matrix_t = mdspan<double, extents_t, layout_left>;
    constexpr double snan = std::numeric_limits<double>::signaling_NaN();

    int m = 3, n = 2;
    std::vector<double> A_mem(m*m, snan);
    std::vector<double> B_mem(m*n);
    std::vector<double> C_mem(m*n, snan);
    std::vector<double> gs_mem(m*n);

    matrix_t A(A_mem.data(), m, m);
    matrix_t B(B_mem.data(), m, n);
    matrix_t C(C_mem.data(), m, n);
    matrix_t gs(gs_mem.data(), m, n);

    // Fill A
    A(0,1) = -2.0;
    A(0,2) = -0.1;
    A(1,2) = 4.5;

    // Fill B
    B(0,0) = -4.4;
    B(0,1) = 1.8;
    B(1,0) = -1.4;
    B(1,1) = 3.4;
    B(2,0) = 1.8;
    B(2,1) = 1.6;

    triangular_matrix_left_product(A, upper_triangle, implicit_unit_diagonal, B, C);

    // Fill GS
    gs(0,0) = -1.78;
    gs(0,1) = -5.16;
    gs(1,0) = 6.7;
    gs(1,1) = 10.6;
    gs(2,0) = 1.8;
    gs(2,1) = 1.6;

    for (ptrdiff_t j = 0; j < n; ++j) {
      for (ptrdiff_t i = 0; i < m; ++i) {
        // FIXME: Choose a more reasonable value for the tolerance
        constexpr double tol = 1e-9;
        EXPECT_NEAR(gs(i,j), C(i,j), tol)
          << "Matrices differ at index ("
          << i << "," << j << ")\n";
      }
    }

    // Check the overwriting version
    triangular_matrix_left_product(A, upper_triangle, implicit_unit_diagonal, B);

    for (ptrdiff_t j = 0; j < n; ++j) {
      for (ptrdiff_t i = 0; i < m; ++i) {
        // FIXME: Choose a more reasonable value for the tolerance
        constexpr double tol = 1e-9;
        EXPECT_NEAR(gs(i,j), B(i,j), tol)
          << "Matrices differ at index ("
          << i << "," << j << ")\n";
      }
    }
  }

TEST(BLAS3_trmm, right_lower_tri_explicit_diag)
  {
    /* C = B * A, where A is triangular mxm */
    using extents_t = dextents<std::size_t, 2>;
    using matrix_t = mdspan<double, extents_t, layout_left>;
    constexpr double snan = std::numeric_limits<double>::signaling_NaN();

    int m = 3, n = 2;
    std::vector<double> A_mem(m*m, snan);
    std::vector<double> B_mem(m*n);
    std::vector<double> C_mem(m*n, snan);
    std::vector<double> gs_mem(m*n);

    matrix_t A(A_mem.data(), m, m);
    matrix_t B(B_mem.data(), n, m);
    matrix_t C(C_mem.data(), n, m);
    matrix_t gs(gs_mem.data(), n, m);

    // Fill A
    A(0,0) = 3.5;
    A(1,0) = -2.0;
    A(1,1) = 1.2;
    A(2,0) = -0.1;
    A(2,1) = 4.5;
    A(2,2) = -1.0;

    // Fill B
    B(0,0) = -4.4;
    B(1,0) = 1.8;
    B(0,1) = -1.4;
    B(1,1) = 3.4;
    B(0,2) = 1.8;
    B(1,2) = 1.6;

    // Fill GS
    gs(0,0) = -12.78;
    gs(1,0) = -0.66;
    gs(0,1) = 6.42;
    gs(1,1) = 11.28;
    gs(0,2) = -1.8;
    gs(1,2) = -1.6;

    // Check the non-overwriting version
    triangular_matrix_right_product(A, lower_triangle, explicit_diagonal, B, C);

    for (ptrdiff_t j = 0; j < m; ++j) {
      for (ptrdiff_t i = 0; i < n; ++i) {
        EXPECT_DOUBLE_EQ(gs(i,j), C(i,j))
          << "Matrices differ at index ("
          << i << "," << j << ")\n";
      }
    }

    // Check the overwriting version
    triangular_matrix_right_product(A, lower_triangle, explicit_diagonal, B);

    for (ptrdiff_t j = 0; j < m; ++j) {
      for (ptrdiff_t i = 0; i < n; ++i) {
        EXPECT_DOUBLE_EQ(gs(i,j), B(i,j))
          << "Matrices differ at index ("
          << i << "," << j << ")\n";
      }
    }
  }

TEST(BLAS3_trmm, right_lower_tri_implicit_diag)
  {
    /* C = A * B, where A is triangular mxm */
    using extents_t = dextents<std::size_t, 2>;
    using matrix_t = mdspan<double, extents_t, layout_left>;
    constexpr double snan = std::numeric_limits<double>::signaling_NaN();

    int m = 3, n = 2;
    std::vector<double> A_mem(m*m, snan);
    std::vector<double> B_mem(m*n);
    std::vector<double> C_mem(m*n, snan);
    std::vector<double> gs_mem(m*n);

    matrix_t A(A_mem.data(), m, m);
    matrix_t B(B_mem.data(), n, m);
    matrix_t C(C_mem.data(), n, m);
    matrix_t gs(gs_mem.data(), n, m);

    // Fill A
    A(1,0) = -2.0;
    A(2,0) = -0.1;
    A(2,1) = 4.5;

    // Fill B
    B(0,0) = -4.4;
    B(1,0) = 1.8;
    B(0,1) = -1.4;
    B(1,1) = 3.4;
    B(0,2) = 1.8;
    B(1,2) = 1.6;

    triangular_matrix_right_product(A, lower_triangle, implicit_unit_diagonal, B, C);

    // Fill GS
    gs(0,0) = -1.78;
    gs(1,0) = -5.16;
    gs(0,1) = 6.7;
    gs(1,1) = 10.6;
    gs(0,2) = 1.8;
    gs(1,2) = 1.6;

    for (ptrdiff_t j = 0; j < m; ++j) {
      for (ptrdiff_t i = 0; i < n; ++i) {
        // FIXME: Choose a more reasonable value for the tolerance
        constexpr double tol = 1e-9;
        EXPECT_NEAR(gs(i,j), C(i,j), tol)
          << "Matrices differ at index ("
          << i << "," << j << ")\n";
      }
    }

    // Check the overwriting version
    triangular_matrix_right_product(A, lower_triangle, implicit_unit_diagonal, B);

    for (ptrdiff_t j = 0; j < m; ++j) {
      for (ptrdiff_t i = 0; i < n; ++i) {
        // FIXME: Choose a more reasonable value for the tolerance
        constexpr double tol = 1e-9;
        EXPECT_NEAR(gs(i,j), B(i,j), tol)
          << "Matrices differ at index ("
          << i << "," << j << ")\n";
      }
    }
  }

TEST(BLAS3_trmm, right_upper_tri_explicit_diag)
  {
    /* C = B*A, where A is triangular mxm */
    using extents_t = dextents<std::size_t, 2>;
    using matrix_t = mdspan<double, extents_t, layout_left>;
    constexpr double snan = std::numeric_limits<double>::signaling_NaN();

    int m = 3, n = 2;
    std::vector<double> A_mem(m*m, snan);
    std::vector<double> B_mem(m*n);
    std::vector<double> C_mem(m*n, snan);
    std::vector<double> gs_mem(m*n);

    matrix_t A(A_mem.data(), m, m);
    matrix_t B(B_mem.data(), n, m);
    matrix_t C(C_mem.data(), n, m);
    matrix_t gs(gs_mem.data(), n, m);

    // Fill A
    A(0,0) = 3.5;
    A(0,1) = -2.0;
    A(1,1) = 1.2;
    A(0,2) = -0.1;
    A(1,2) = 4.5;
    A(2,2) = -1.0;

    // Fill B
    B(0,0) = -4.4;
    B(1,0) = 1.8;
    B(0,1) = -1.4;
    B(1,1) = 3.4;
    B(0,2) = 1.8;
    B(1,2) = 1.6;

    // Fill GS
    gs(0,0) = -15.4;
    gs(1,0) = 6.3;
    gs(0,1) = 7.12;
    gs(1,1) = 0.48;
    gs(0,2) = -7.66;
    gs(1,2) = 13.52;

    // Check the non-overwriting version
    triangular_matrix_right_product(A, upper_triangle, explicit_diagonal, B, C);

    for (ptrdiff_t j = 0; j < m; ++j) {
      for (ptrdiff_t i = 0; i < n; ++i) {
        EXPECT_DOUBLE_EQ(gs(i,j), C(i,j))
          << "Matrices differ at index ("
          << i << "," << j << ")\n";
      }
    }

    // Check the overwriting version
    triangular_matrix_right_product(A, upper_triangle, explicit_diagonal, B);

    for (ptrdiff_t j = 0; j < m; ++j) {
      for (ptrdiff_t i = 0; i < n; ++i) {
        EXPECT_DOUBLE_EQ(gs(i,j), B(i,j))
          << "Matrices differ at index ("
          << i << "," << j << ")\n";
      }
    }
  }

TEST(BLAS3_trmm, right_upper_tri_implicit_diag)
  {
    /* C = B * A, where A is triangular mxm */
    using extents_t = dextents<std::size_t, 2>;
    using matrix_t = mdspan<double, extents_t, layout_left>;
    constexpr double snan = std::numeric_limits<double>::signaling_NaN();

    int m = 3, n = 2;
    std::vector<double> A_mem(m*m, snan);
    std::vector<double> B_mem(m*n);
    std::vector<double> C_mem(m*n, snan);
    std::vector<double> gs_mem(m*n);

    matrix_t A(A_mem.data(), m, m);
    matrix_t B(B_mem.data(), n, m);
    matrix_t C(C_mem.data(), n, m);
    matrix_t gs(gs_mem.data(), n, m);

    // Fill A
    A(0,1) = -2.0;
    A(0,2) = -0.1;
    A(1,2) = 4.5;

    // Fill B
    B(0,0) = -4.4;
    B(1,0) = 1.8;
    B(0,1) = -1.4;
    B(1,1) = 3.4;
    B(0,2) = 1.8;
    B(1,2) = 1.6;

    triangular_matrix_right_product(A, upper_triangle, implicit_unit_diagonal, B, C);

    // Fill GS
    gs(0,0) = -4.4;
    gs(1,0) = 1.8;
    gs(0,1) = 7.4;
    gs(1,1) = -0.2;
    gs(0,2) = -4.06;
    gs(1,2) = 16.72;

    for (ptrdiff_t j = 0; j < m; ++j) {
      for (ptrdiff_t i = 0; i < n; ++i) {
        // FIXME: Choose a more reasonable value for the tolerance
        constexpr double tol = 1e-9;
        EXPECT_NEAR(gs(i,j), C(i,j), tol)
          << "Matrices differ at index ("
          << i << "," << j << ")\n";
      }
    }

    // Check the overwriting version
    triangular_matrix_right_product(A, upper_triangle, implicit_unit_diagonal, B);

    for (ptrdiff_t j = 0; j < m; ++j) {
      for (ptrdiff_t i = 0; i < n; ++i) {
        // FIXME: Choose a more reasonable value for the tolerance
        constexpr double tol = 1e-9;
        EXPECT_NEAR(gs(i,j), B(i,j), tol)
          << "Matrices differ at index ("
          << i << "," << j << ")\n";
      }
    }
  }

} // end anonymous namespace
