#include "./gtest_fixtures.hpp"

#include <experimental/linalg>
#include <iostream>

namespace {
  constexpr std::size_t num_rows_A = 3;
  constexpr std::size_t num_cols_A = 3;
  constexpr double storage_A[] =
    {8., 0., 0.,
     2., 8., 0.,
     1., 2., 8.};
  constexpr std::size_t num_rows_B = 4;
  constexpr std::size_t num_cols_B = 3;
  constexpr double storage_B[] =
    {1.,  2.,  3.,
     4.,  5.,  6.,
     7.,  8.,  9.,
     10., 11., 12.};
  constexpr double storage_B_times_A[] =
    {15.,  22.,  24.,
     48.,  52.,  48.,
     81.,  82.,  72.,
     114., 112., 96.};
  constexpr double storage_B_times_inv_A[] =
    {0.0390625, 0.15625, 0.375,
     0.296875 , 0.4375 , 0.75,
     0.5546875, 0.71875, 1.125,
     0.8125   , 1.     , 1.5};

  template<class IndexType, class Layout>
  void fill_from_layout_right_storage(
    mdspan<double, dextents<IndexType, 2>, Layout> out,
    const double* const in_storage,
    const std::size_t num_rows,
    const std::size_t num_cols)
  {
    mdspan<const double, dextents<std::size_t, 2>, layout_right> in(in_storage, num_rows, num_cols);
    for(std::size_t i = 0; i < num_rows; ++i) {
      for(std::size_t j = 0; j < num_cols; ++j) {
	out(i,j) = in(i,j);
      }
    }
  }

  // Regression test for https://github.com/kokkos/stdBLAS/issues/244 .
  // It will fail if the j loop (mentioned in the bug) counts up instead of down.
  template<class IndexType, class Layout>
  void test_tsrm_lower_triangular_right_side()
  {
    std::vector<double> vec_A(num_rows_A * num_cols_A);
    std::vector<double> vec_B(num_rows_B * num_cols_B);
    const std::size_t num_rows_X = num_rows_B;
    const std::size_t num_cols_X = num_cols_B;
    std::vector<double> vec_X(num_rows_X * num_cols_X);

    mdspan<double, dextents<IndexType, 2>, Layout> A(vec_A.data(), num_rows_A, num_cols_A);
    mdspan<double, dextents<IndexType, 2>, Layout> B_nonconst(vec_B.data(), num_rows_B, num_cols_B);
    mdspan<double, dextents<IndexType, 2>, Layout> X(vec_X.data(), num_rows_X, num_cols_X);

    fill_from_layout_right_storage<IndexType, Layout>(A, storage_A, num_rows_A, num_cols_A);
    fill_from_layout_right_storage<IndexType, Layout>(B_nonconst, storage_B, num_rows_B, num_cols_B);
    mdspan<const double, dextents<IndexType, 2>, Layout> B = B_nonconst;

    using ::std::experimental::linalg::explicit_diagonal;
    using ::std::experimental::linalg::lower_triangle;
    using ::std::experimental::linalg::right_side;
    using ::std::experimental::linalg::triangular_matrix_matrix_solve;
    triangular_matrix_matrix_solve(A, lower_triangle, explicit_diagonal, right_side, B, X);

    mdspan<const double, dextents<IndexType, 2>, layout_right>
      B_times_inv_A(storage_B_times_inv_A, num_rows_B, num_cols_A);

    for(IndexType r = 0; r < IndexType(num_rows_B); ++r) {
      for(IndexType c = 0; c < IndexType(num_cols_A); ++c) {
	// We chose the values in A and B so that triangular
	// solve could compute them without rounding error.
	EXPECT_EQ( X(r,c), B_times_inv_A(r,c) );
      }
    }
  }

  TEST(BLAS3_trsm, double_size_t_layout_right )
  {
    test_tsrm_lower_triangular_right_side< ::std::size_t, layout_right >();
  }

  TEST(BLAS3_trsm, double_int_layout_left )
  {
    test_tsrm_lower_triangular_right_side< int, layout_left >();
  }

} // end anonymous namespace
