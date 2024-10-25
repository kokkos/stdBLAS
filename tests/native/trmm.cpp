#include "./gtest_fixtures.hpp"
#include <iostream>

namespace {
  using LinearAlgebra::explicit_diagonal;
  using LinearAlgebra::implicit_unit_diagonal;
  using LinearAlgebra::lower_triangle;
  using LinearAlgebra::matrix_product;
  using LinearAlgebra::transposed;
  using LinearAlgebra::upper_triangle;
  using std::cout;
  using std::endl;

  template<bool isLeft, typename Triangle, typename Diagonal>
  void test_triangular_matrix_product()
  {
    /* C = A * B, where A is triangular mxm */
    using extents_t = dextents<std::size_t, 2>;
    using matrix_t = mdspan<double, extents_t, layout_left>;
    constexpr double snan = std::numeric_limits<double>::signaling_NaN();

    int m = 3, n = 2;
    std::vector<double> A_mem(m*m, snan);
    std::vector<double> B_mem(m*n);
    std::vector<double> E_mem(m*n);
    std::vector<double> C_mem(m*n, snan);
    std::vector<double> gs_mem(m*n);

    matrix_t A(A_mem.data(), m, m);
    matrix_t B(B_mem.data(), (isLeft)?m:n, (isLeft)?n:m);
    matrix_t E(E_mem.data(), (isLeft)?m:n, (isLeft)?n:m);
    matrix_t C(C_mem.data(), (isLeft)?m:n, (isLeft)?n:m);
    matrix_t gs(gs_mem.data(), m, n);

    // Fill A
    if(std::is_same_v<Triangle, decltype(lower_triangle)>) {
      if(std::is_same_v<Diagonal, decltype(explicit_diagonal)>) {
        A(0,0) = 3.5;
        A(1,1) = 1.2;
        A(2,2) = -1.0;
      }

      A(1,0) = -2.0;
      A(2,0) = -0.1;
      A(2,1) = 4.5;
    }
    else {
      if(std::is_same_v<Diagonal, decltype(explicit_diagonal)>) {
        A(0,0) = 3.5;
        A(1,1) = 1.2;
        A(2,2) = -1.0;
      }

      A(0,1) = -2.0;
      A(0,2) = -0.1;
      A(1,2) = 4.5;
    }

    // Fill B
    if(isLeft) {
      B(0,0) = -4.4;
      B(0,1) = 1.8;
      B(1,0) = -1.4;
      B(1,1) = 3.4;
      B(2,0) = 1.8;
      B(2,1) = 1.6;
    }
    else {
      B(0,0) = -4.4;
      B(1,0) = 1.8;
      B(0,1) = -1.4;
      B(1,1) = 3.4;
      B(0,2) = 1.8;
      B(1,2) = 1.6;
    }

    for (ptrdiff_t j = 0; j < (isLeft?n:m); ++j) {
      for (ptrdiff_t i = 0; i < (isLeft?m:n); ++i) {
        E(i,j) = B(i,j);
      }
    }

    // Fill GS
    if(std::is_same_v<Diagonal, decltype(explicit_diagonal)>) {
        if(std::is_same_v<Triangle, decltype(lower_triangle)> == isLeft) {
          gs(0,0) = -15.4;
          gs(0,1) = 6.3;
          gs(1,0) = 7.12;
          gs(1,1) = 0.48;
          gs(2,0) = -7.66;
          gs(2,1) = 13.52;
        }
        else {
          gs(0,0) = -12.78;
          gs(0,1) = -0.66;
          gs(1,0) = 6.42;
          gs(1,1) = 11.28;
          gs(2,0) = -1.8;
          gs(2,1) = -1.6;
        }
    }
    else {
        if(std::is_same_v<Triangle, decltype(lower_triangle)> == isLeft) {
          gs(0,0) = -4.4;
          gs(0,1) = 1.8; 
          gs(1,0) = 7.4; 
          gs(1,1) = -0.2;
          gs(2,0) = -4.06;
          gs(2,1) = 16.72;
        }
        else {
          gs(0,0) = -1.78;
          gs(0,1) = -5.16;
          gs(1,0) = 6.7;
          gs(1,1) = 10.6;
          gs(2,0) = 1.8;
          gs(2,1) = 1.6;
        }
    }

    // Check the Updating version
    if(isLeft) {
      triangular_matrix_product(A, Triangle{}, Diagonal{}, B, E, C);
    }
    else {
      triangular_matrix_product(B, A, Triangle{}, Diagonal{}, E, C);
    }

    for (ptrdiff_t j = 0; j < (isLeft?n:m); ++j) {
      for (ptrdiff_t i = 0; i < (isLeft?m:n); ++i) {
        constexpr double tol = 1e-9;
        if(isLeft) {
          EXPECT_NEAR(gs(i,j)+E(i,j), C(i,j), tol)
            << "Matrices differ at index ("
            << i << "," << j << ")\n";
        }
        else {
          EXPECT_NEAR(gs(j,i)+E(i,j), C(i,j), tol)
            << "Matrices differ at index ("
            << i << "," << j << ")\n";
        }
      }
    }

    // Check the non-overwriting version
    if(isLeft) {
      triangular_matrix_product(A, Triangle{}, Diagonal{}, B, C);
    }
    else {
      triangular_matrix_product(B, A, Triangle{}, Diagonal{}, C);
    }

    for (ptrdiff_t j = 0; j < (isLeft?n:m); ++j) {
      for (ptrdiff_t i = 0; i < (isLeft?m:n); ++i) {
        constexpr double tol = 1e-9;
        if(isLeft) {
          EXPECT_NEAR(gs(i,j), C(i,j), tol)
            << "Matrices differ at index ("
            << i << "," << j << ")\n";
        }
        else {
          EXPECT_NEAR(gs(j,i), C(i,j), tol)
            << "Matrices differ at index ("
            << i << "," << j << ")\n";
        }
      }
    }

    // Check the overwriting version
    if(isLeft) {
      triangular_matrix_left_product(A, Triangle{}, Diagonal{}, B);
    }
    else {
      triangular_matrix_right_product(A, Triangle{}, Diagonal{}, B);
    }

    for (ptrdiff_t j = 0; j < (isLeft?n:m); ++j) {
      for (ptrdiff_t i = 0; i < (isLeft?m:n); ++i) {
        constexpr double tol = 1e-9;
        if(isLeft) {
          EXPECT_NEAR(gs(i,j), B(i,j), tol)
            << "Matrices differ at index ("
            << i << "," << j << ")\n";
        }
        else {
          EXPECT_NEAR(gs(j,i), B(i,j), tol)
            << "Matrices differ at index ("
            << i << "," << j << ")\n";
        }
      }
    }
  }


  TEST(BLAS3_trmm, left_lower_tri_explicit_diag)
  {
    test_triangular_matrix_product<true, decltype(lower_triangle), decltype(explicit_diagonal)>();
  }


  TEST(BLAS3_trmm, left_lower_tri_implicit_diag)
  {
    test_triangular_matrix_product<true, decltype(lower_triangle), decltype(implicit_unit_diagonal)>();
  }

  TEST(BLAS3_trmm, left_upper_tri_explicit_diag)
  {
    test_triangular_matrix_product<true, decltype(upper_triangle), decltype(explicit_diagonal)>();
  }

  TEST(BLAS3_trmm, left_upper_tri_implicit_diag)
  {
    test_triangular_matrix_product<true, decltype(upper_triangle), decltype(implicit_unit_diagonal)>();
  }

  TEST(BLAS3_trmm, right_lower_tri_explicit_diag)
  {
    test_triangular_matrix_product<false, decltype(lower_triangle), decltype(explicit_diagonal)>();
  }

  TEST(BLAS3_trmm, right_lower_tri_implicit_diag)
  {
    test_triangular_matrix_product<false, decltype(lower_triangle), decltype(implicit_unit_diagonal)>();
  }

  TEST(BLAS3_trmm, right_upper_tri_explicit_diag)
  {
    test_triangular_matrix_product<false, decltype(upper_triangle), decltype(explicit_diagonal)>();
  }

  TEST(BLAS3_trmm, right_upper_tri_implicit_diag)
  {
    test_triangular_matrix_product<false, decltype(upper_triangle), decltype(implicit_unit_diagonal)>();
  }

} // end anonymous namespace
