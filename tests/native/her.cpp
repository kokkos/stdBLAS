#include "./gtest_fixtures.hpp"

namespace {
  using LinearAlgebra::lower_triangle;
  using LinearAlgebra::hermitian_matrix_rank_1_update;
  using LinearAlgebra::upper_triangle;

  // A = [-1.0  -2.0  -4.0]
  //     [-2.0  -3.0  -5.0]
  //     [-4.0  -5.0  -6.0]
  //
  // x = [  2.0 + 7.0i]
  //     [  5.0]
  //     [ 11.0]
  //
  // x x^H = [53.0          10.0 + 35.0i   22.0 + 77.0i]
  //         [10.0 - 35.0i  25.0           55.0        ]
  //         [22.0 - 77.0i  55.0          121.0        ]
  //
  // A + x x^H = [52.0           8.0 + 35.0i   18.0 + 77.0i]
  //             [ 8.0 - 35.0i  22.0           50.0]
  //             [18.0 - 77.0i  50.0          115.0]
  class her_test_problem {
  public:
    using V = std::complex<double>;

    std::vector<V> A_original{
      V(-1.0, 0.0), V(-2.0, 0.0),  V(-4.0, 0.0),
      V(-2.0, 0.0), V(-3.0, 0.0),  V(-5.0, 0.0),
      V(-4.0, 0.0), V(-5.0, 0.0),  V(-6.0, 0.0)
    };

    std::vector<V> A{
      V(-1.0, 0.0), V(-2.0, 0.0),  V(-4.0, 0.0),
      V(-2.0, 0.0), V(-3.0, 0.0),  V(-5.0, 0.0),
      V(-4.0, 0.0), V(-5.0, 0.0),  V(-6.0, 0.0)
    };

    std::vector<V> x{
      V( 2.0, 7.0),
      V( 5.0, 0.0),
      V(11.0, 0.0)
    };

    std::vector<V> A_plus_x_xH = {
      V(52.0,   0.0), V( 8.0, 35.0), V( 18.0, 77.0),
      V( 8.0, -35.0), V(22.0,  0.0), V( 50.0,  0.0),
      V(18.0, -77.0), V(50.0,  0.0), V(115.0,  0.0)
    };

#if defined(LINALG_FIX_RANK_UPDATES)
    std::vector<V> x_xH = {
      V(53.0,   0.0), V(10.0, 35.0), V( 22.0, 77.0),
      V(10.0, -35.0), V(25.0,  0.0), V( 55.0,  0.0),
      V(22.0, -77.0), V(55.0,  0.0), V(121.0,  0.0)
    };

    std::vector<V> two_x_xH = {
      V(106.0,   0.0),  V( 20.0, 70.0), V( 44.0, 154.0),
      V( 20.0,  -70.0), V( 50.0,  0.0), V(110.0,   0.0),
      V( 44.0, -154.0), V(110.0,  0.0), V(242.0,   0.0)
    };

    std::vector<V> A_plus_two_x_xH = {
      V(105.0,    0.0), V( 18.0, 70.0), V( 40.0, 154.0),
      V( 18.0,  -70.0), V( 47.0,  0.0), V(105.0,   0.0),
      V( 40.0, -154.0), V(105.0,  0.0), V(236.0,   0.0)
    };
#endif // LINALG_FIX_RANK_UPDATES

    using A_type = mdspan<V, extents<int, 3, 3>>;
    using const_A_type = mdspan<const V, extents<int, 3, 3>>;
    using x_type = mdspan<const V, extents<int, 3>>;
    using result_type = mdspan<const V, extents<int, 3, 3>>;

    A_type A_view() {
      return A_type{A.data()};
    }

    const_A_type A_original_view() const {
      return const_A_type{A_original.data()};
    }

    x_type x_view() const {
      return x_type{x.data()};
    }

    result_type A_plus_x_xH_view() const {
      return result_type{A_plus_x_xH.data()};
    }

#if defined(LINALG_FIX_RANK_UPDATES)
    result_type x_xH_view() const {
      return result_type{x_xH.data()};
    }

    result_type two_x_xH_view() const {
      return result_type{two_x_xH.data()};
    }

    result_type A_plus_two_x_xH_view() const {
      return result_type{A_plus_two_x_xH.data()};
    }
#endif // LINALG_FIX_RANK_UPDATES
  };

  // This also serves as a regression test for ambiguous overloads of
  // hermitian_matrix_rank_1_update (related to
  // https://github.com/kokkos/stdBLAS/issues/261 ).
  //
  // The reference implementation needs to implement all constraints
  // of hermitian_matrix_rank_1_update in order to disambiguate
  // overloads.
  TEST(BLAS3_her, upper_triangle)
  {
    {
      her_test_problem problem;
      mdspan A = problem.A_view();
      mdspan A_original = problem.A_original_view();
      mdspan x = problem.x_view();
      mdspan A_plus_x_xH = problem.A_plus_x_xH_view();
      const char what[] = " triangle of A = A + 1.0 x x^H (upper)";

#if defined(LINALG_FIX_RANK_UPDATES)
      hermitian_matrix_rank_1_update(1.0, x, A, A, upper_triangle);
#else
      hermitian_matrix_rank_1_update(1.0, x, A, upper_triangle);
#endif // LINALG_FIX_RANK_UPDATES
      for (int row = 0; row < A.extent(0); ++row) {
        // Lower triangle is unchanged.
        for (int col = 0; col < row; ++col) {
          EXPECT_EQ(A(row, col), A_original(row, col))
            << "A(" << row << "," << col << ") is wrong for "
            << "lower" << what;
        }
        // Upper triangle is changed.
        for (int col = row; col < A.extent(1); ++col) {
          EXPECT_EQ(A(row, col), A_plus_x_xH(row, col))
            << "A(" << row << "," << col << ") is wrong for "
            << "upper" << what;
        }
      }
    }

    {
      her_test_problem problem;
      mdspan A = problem.A_view();
      mdspan A_original = problem.A_original_view();
      mdspan x = problem.x_view();
      mdspan A_plus_x_xH = problem.A_plus_x_xH_view();
      const char what[] = " triangle of A = A + x x^H (upper)";

#if defined(LINALG_FIX_RANK_UPDATES)
      hermitian_matrix_rank_1_update(x, A, A, upper_triangle);
#else
      hermitian_matrix_rank_1_update(x, A, upper_triangle);
#endif // LINALG_FIX_RANK_UPDATES
      for (int row = 0; row < A.extent(0); ++row) {
        // Lower triangle is unchanged.
        for (int col = 0; col < row; ++col) {
          EXPECT_EQ(A(row, col), A_original(row, col))
            << "A(" << row << "," << col << ") is wrong for "
            << "lower" << what;
        }
        // Upper triangle is changed.
        for (int col = row; col < A.extent(1); ++col) {
          EXPECT_EQ(A(row, col), A_plus_x_xH(row, col))
            << "A(" << row << "," << col << ") is wrong for "
            << "upper" << what;
        }
      }
    }

#if defined(LINALG_FIX_RANK_UPDATES)
    {
      her_test_problem problem;
      mdspan A = problem.A_view();
      mdspan A_original = problem.A_original_view();
      mdspan x = problem.x_view();
      mdspan x_xH = problem.x_xH_view();
      const char what[] = " triangle of A = 1.0 x x^H (upper)";

      hermitian_matrix_rank_1_update(1.0, x, A, upper_triangle);
      for (int row = 0; row < A.extent(0); ++row) {
        // Lower triangle is unchanged.
        for (int col = 0; col < row; ++col) {
          EXPECT_EQ(A(row, col), A_original(row, col))
            << "A(" << row << "," << col << ") is wrong for "
            << "lower" << what;
        }
        // Upper triangle is changed.
        for (int col = row; col < A.extent(1); ++col) {
          EXPECT_EQ(A(row, col), x_xH(row, col))
            << "A(" << row << "," << col << ") is wrong for "
            << "upper" << what;
        }
      }
    }

    {
      her_test_problem problem;
      mdspan A = problem.A_view();
      mdspan A_original = problem.A_original_view();
      mdspan x = problem.x_view();
      mdspan A_plus_x_xH = problem.A_plus_x_xH_view();
      const char what[] = " triangle of A = 2.0 x x^H (upper)";

      hermitian_matrix_rank_1_update(2.0, x, A, upper_triangle);
      mdspan two_x_xH = problem.two_x_xH_view();
      for (int row = 0; row < A.extent(0); ++row) {
        // Lower triangle is unchanged.
        for (int col = 0; col < row; ++col) {
          EXPECT_EQ(A(row, col), A_original(row, col))
            << "A(" << row << "," << col << ") is wrong for "
            << "lower" << what;
        }
        // Upper triangle is changed.
        for (int col = row; col < A.extent(1); ++col) {
          EXPECT_EQ(A(row, col), two_x_xH(row, col))
            << "A(" << row << "," << col << ") is wrong for "
            << "upper" << what;
        }
      }
    }

    {
      her_test_problem problem;
      mdspan A = problem.A_view();
      mdspan A_original = problem.A_original_view();
      mdspan x = problem.x_view();
      mdspan A_plus_two_x_xH = problem.A_plus_two_x_xH_view();
      const char what[] = " triangle of A = A + 2.0 x x^H (upper)";

      hermitian_matrix_rank_1_update(2.0, x, A, A, upper_triangle);
      for (int row = 0; row < A.extent(0); ++row) {
        // Lower triangle is unchanged.
        for (int col = 0; col < row; ++col) {
          EXPECT_EQ(A(row, col), A_original(row, col))
            << "A(" << row << "," << col << ") is wrong for "
            << "lower" << what;
        }
        // Upper triangle is changed.
        for (int col = row; col < A.extent(1); ++col) {
          EXPECT_EQ(A(row, col), A_plus_two_x_xH(row, col))
            << "A(" << row << "," << col << ") is wrong for "
            << "upper" << what;
        }
      }
    }
#endif // LINALG_FIX_RANK_UPDATES
  }

  TEST(BLAS3_her, lower_triangle)
  {
    {
      her_test_problem problem;
      mdspan A = problem.A_view();
      mdspan A_original = problem.A_original_view();
      mdspan x = problem.x_view();
      mdspan A_plus_x_xH = problem.A_plus_x_xH_view();
      const char what[] = " triangle of A = A + 1.0 x x^H (lower)";

#if defined(LINALG_FIX_RANK_UPDATES)
      hermitian_matrix_rank_1_update(1.0, x, A, A, lower_triangle);
#else
      hermitian_matrix_rank_1_update(1.0, x, A, lower_triangle);
#endif // LINALG_FIX_RANK_UPDATES
      for (int row = 0; row < A.extent(0); ++row) {
        // Lower triangle is changed.
        for (int col = 0; col <= row; ++col) {
          EXPECT_EQ(A(row, col), A_plus_x_xH(row, col))
            << "A(" << row << "," << col << ") is wrong for "
            << "lower" << what;
        }
        // Upper triangle is unchanged.
        for (int col = row+1; col < A.extent(1); ++col) {
          EXPECT_EQ(A(row, col), A_original(row, col))
            << "A(" << row << "," << col << ") is wrong for "
            << "upper" << what;
        }
      }
    }

    {
      her_test_problem problem;
      mdspan A = problem.A_view();
      mdspan A_original = problem.A_original_view();
      mdspan x = problem.x_view();
      mdspan A_plus_x_xH = problem.A_plus_x_xH_view();
      const char what[] = " triangle of A = A + x x^H (lower)";

#if defined(LINALG_FIX_RANK_UPDATES)
      hermitian_matrix_rank_1_update(x, A, A, lower_triangle);
#else
      hermitian_matrix_rank_1_update(x, A, lower_triangle);
#endif // LINALG_FIX_RANK_UPDATES
      for (int row = 0; row < A.extent(0); ++row) {
        // Lower triangle is changed.
        for (int col = 0; col <= row; ++col) {
          EXPECT_EQ(A(row, col), A_plus_x_xH(row, col))
            << "A(" << row << "," << col << ") is wrong for "
            << "lower" << what;
        }
        // Upper triangle is unchanged.
        for (int col = row+1; col < A.extent(1); ++col) {
          EXPECT_EQ(A(row, col), A_original(row, col))
            << "A(" << row << "," << col << ") is wrong for "
            << "upper" << what;
        }
      }
    }

#if defined(LINALG_FIX_RANK_UPDATES)
    {
      her_test_problem problem;
      mdspan A = problem.A_view();
      mdspan A_original = problem.A_original_view();
      mdspan x = problem.x_view();
      mdspan x_xH = problem.x_xH_view();
      const char what[] = " triangle of A = x x^H (lower)";

      hermitian_matrix_rank_1_update(x, A, lower_triangle);
      for (int row = 0; row < A.extent(0); ++row) {
        // Lower triangle is changed.
        for (int col = 0; col <= row; ++col) {
          EXPECT_EQ(A(row, col), x_xH(row, col))
            << "A(" << row << "," << col << ") is wrong for "
            << "lower" << what;
        }
        // Upper triangle is unchanged.
        for (int col = row+1; col < A.extent(1); ++col) {
          EXPECT_EQ(A(row, col), A_original(row, col))
            << "A(" << row << "," << col << ") is wrong for "
            << "upper" << what;
        }
      }
    }

    {
      her_test_problem problem;
      mdspan A = problem.A_view();
      mdspan A_original = problem.A_original_view();
      mdspan x = problem.x_view();
      mdspan x_xH = problem.x_xH_view();
      const char what[] = " triangle of A = 1.0 x x^H (lower)";

      hermitian_matrix_rank_1_update(1.0, x, A, lower_triangle);
      for (int row = 0; row < A.extent(0); ++row) {
        // Lower triangle is changed.
        for (int col = 0; col <= row; ++col) {
          EXPECT_EQ(A(row, col), x_xH(row, col))
            << "A(" << row << "," << col << ") is wrong for "
            << "lower" << what;
        }
        // Upper triangle is unchanged.
        for (int col = row+1; col < A.extent(1); ++col) {
          EXPECT_EQ(A(row, col), A_original(row, col))
            << "A(" << row << "," << col << ") is wrong for "
            << "upper" << what;
        }
      }
    }

    {
      her_test_problem problem;
      mdspan A = problem.A_view();
      mdspan A_original = problem.A_original_view();
      mdspan x = problem.x_view();
      mdspan two_x_xH = problem.two_x_xH_view();
      const char what[] = " triangle of A = 2.0 x x^H (lower)";

      hermitian_matrix_rank_1_update(2.0, x, A, lower_triangle);
      for (int row = 0; row < A.extent(0); ++row) {
        // Lower triangle is changed.
        for (int col = 0; col <= row; ++col) {
          EXPECT_EQ(A(row, col), two_x_xH(row, col))
            << "A(" << row << "," << col << ") is wrong for "
            << "lower" << what;
        }
        // Upper triangle is unchanged.
        for (int col = row+1; col < A.extent(1); ++col) {
          EXPECT_EQ(A(row, col), A_original(row, col))
            << "A(" << row << "," << col << ") is wrong for "
            << "upper" << what;
        }
      }
    }

    {
      her_test_problem problem;
      mdspan A = problem.A_view();
      mdspan A_original = problem.A_original_view();
      mdspan x = problem.x_view();
      mdspan A_plus_two_x_xH = problem.A_plus_two_x_xH_view();
      const char what[] = " triangle of A = A + 2.0 x x^H (lower)";

      hermitian_matrix_rank_1_update(2.0, x, A, A, lower_triangle);
      for (int row = 0; row < A.extent(0); ++row) {
        // Lower triangle is changed.
        for (int col = 0; col <= row; ++col) {
          EXPECT_EQ(A(row, col), A_plus_two_x_xH(row, col))
            << "A(" << row << "," << col << ") is wrong for "
            << "lower" << what;
        }
        // Upper triangle is unchanged.
        for (int col = row+1; col < A.extent(1); ++col) {
          EXPECT_EQ(A(row, col), A_original(row, col))
            << "A(" << row << "," << col << ") is wrong for "
            << "upper" << what;
        }
      }
    }
#endif // LINALG_FIX_RANK_UPDATES
  }
} // end anonymous namespace
