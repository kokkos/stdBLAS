#include "./gtest_fixtures.hpp"

namespace {
  using LinearAlgebra::lower_triangle;
  using LinearAlgebra::symmetric_matrix_rank_1_update;
  using LinearAlgebra::upper_triangle;

  // A = [-1.0  -2.0  -4.0]
  //     [-2.0  -3.0  -5.0]
  //     [-4.0  -5.0  -6.0]
  //
  // x = [  2.0]
  //     [  5.0]
  //     [ 11.0]
  //
  // x x^T = [4.0   10.0  22.0]
  //         [10.0  25.0  55.0]
  //         [22.0  55.0 121.0]
  //
  // 2.0 x x^T = [ 8.0  20.0  44.0]
  //             [20.0  50.0 110.0]
  //             [44.0 110.0 242.0]
  //
  // A + x x^T = [3.0   8.0  18.0]
  //             [8.0  22.0  50.0]
  //             [18.0 50.0 115.0]
  //
  // A + 2.0 x x^T = [ 7.0  18.0  40.0]
  //                 [18.0  47.0 105.0]
  //                 [40.0 105.0 236.0]
  class syr_test_problem {
  public:
    using V = double;

    std::vector<V> A_original{
      V(-1.0), V(-2.0), V(-4.0),
      V(-2.0), V(-3.0), V(-5.0),
      V(-4.0), V(-5.0), V(-6.0)
    };

    std::vector<V> A{
      V(-1.0), V(-2.0), V(-4.0),
      V(-2.0), V(-3.0), V(-5.0),
      V(-4.0), V(-5.0), V(-6.0)
    };

    std::vector<V> x{
      V( 2.0),
      V( 5.0),
      V(11.0)
    };

    std::vector<V> A_plus_x_xT = {
      V( 3.0), V( 8.0), V( 18.0),
      V( 8.0), V(22.0), V( 50.0),
      V(18.0), V(50.0), V(115.0)
    };

#if defined(LINALG_FIX_RANK_UPDATES)
    std::vector<V> x_xT = {
      V( 4.0), V(10.0), V( 22.0),
      V(10.0), V(25.0), V( 55.0),
      V(22.0), V(55.0), V(121.0)
    };

    std::vector<V> two_x_xT = {
      V( 8.0), V( 20.0), V( 44.0),
      V(20.0), V( 50.0), V(110.0),
      V(44.0), V(110.0), V(242.0)
    };

    std::vector<V> A_plus_two_x_xT = {
      V( 7.0), V( 18.0), V( 40.0),
      V(18.0), V( 47.0), V(105.0),
      V(40.0), V(105.0), V(236.0)
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

    result_type A_plus_x_xT_view() const {
      return result_type{A_plus_x_xT.data()};
    }

#if defined(LINALG_FIX_RANK_UPDATES)
    result_type x_xT_view() const {
      return result_type{x_xT.data()};
    }

    result_type two_x_xT_view() const {
      return result_type{two_x_xT.data()};
    }

    result_type A_plus_two_x_xT_view() const {
      return result_type{A_plus_two_x_xT.data()};
    }
#endif // LINALG_FIX_RANK_UPDATES
  };
  
  // This also serves as a regression test for ambiguous overloads of
  // symmetric_matrix_rank_1_update (related to
  // https://github.com/kokkos/stdBLAS/issues/261).
  //
  // The reference implementation needs to implement all constraints
  // of symmetric_matrix_rank_1_update in order to disambiguate
  // overloads.
  TEST(BLAS3_syr, upper_triangle)
  {
    {
      syr_test_problem problem;
      mdspan A = problem.A_view();
      mdspan A_original = problem.A_original_view();
      mdspan x = problem.x_view();
      mdspan A_plus_x_xT = problem.A_plus_x_xT_view();
      const char what[] = " triangle of A = A + 1.0 x x^T (upper)";

#if defined(LINALG_FIX_RANK_UPDATES)
      symmetric_matrix_rank_1_update(1.0, x, A, A, upper_triangle);
#else
      symmetric_matrix_rank_1_update(1.0, x, A, upper_triangle);
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
          EXPECT_EQ(A(row, col), A_plus_x_xT(row, col))
            << "A(" << row << "," << col << ") is wrong for "
            << "upper" << what;
        }
      }
    }

    {
      syr_test_problem problem;
      mdspan A = problem.A_view();
      mdspan A_original = problem.A_original_view();
      mdspan x = problem.x_view();
      mdspan A_plus_x_xT = problem.A_plus_x_xT_view();
      const char what[] = " triangle of A = A + x x^T (upper)";

#if defined(LINALG_FIX_RANK_UPDATES)
      symmetric_matrix_rank_1_update(x, A, A, upper_triangle);
#else
      symmetric_matrix_rank_1_update(x, A, upper_triangle);
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
          EXPECT_EQ(A(row, col), A_plus_x_xT(row, col))
            << "A(" << row << "," << col << ") is wrong for "
            << "upper" << what;
        }
      }
    }

#if defined(LINALG_FIX_RANK_UPDATES)
    {
      syr_test_problem problem;
      mdspan A = problem.A_view();
      mdspan A_original = problem.A_original_view();
      mdspan x = problem.x_view();
      mdspan x_xT = problem.x_xT_view();
      const char what[] = " triangle of A = 1.0 x x^T (upper)";

      symmetric_matrix_rank_1_update(1.0, x, A, upper_triangle);
      for (int row = 0; row < A.extent(0); ++row) {
        // Lower triangle is unchanged.
        for (int col = 0; col < row; ++col) {
          EXPECT_EQ(A(row, col), A_original(row, col))
            << "A(" << row << "," << col << ") is wrong for "
            << "lower" << what;
        }
        // Upper triangle is changed.
        for (int col = row; col < A.extent(1); ++col) {
          EXPECT_EQ(A(row, col), x_xT(row, col))
            << "A(" << row << "," << col << ") is wrong for "
            << "upper" << what;
        }
      }
    }

    {
      syr_test_problem problem;
      mdspan A = problem.A_view();
      mdspan A_original = problem.A_original_view();
      mdspan x = problem.x_view();
      mdspan A_plus_x_xT = problem.A_plus_x_xT_view();
      const char what[] = " triangle of A = 2.0 x x^T (upper)";

      symmetric_matrix_rank_1_update(2.0, x, A, upper_triangle);
      mdspan two_x_xT = problem.two_x_xT_view();
      for (int row = 0; row < A.extent(0); ++row) {
        // Lower triangle is unchanged.
        for (int col = 0; col < row; ++col) {
          EXPECT_EQ(A(row, col), A_original(row, col))
            << "A(" << row << "," << col << ") is wrong for "
            << "lower" << what;
        }
        // Upper triangle is changed.
        for (int col = row; col < A.extent(1); ++col) {
          EXPECT_EQ(A(row, col), two_x_xT(row, col))
            << "A(" << row << "," << col << ") is wrong for "
            << "upper" << what;
        }
      }
    }

    {
      syr_test_problem problem;
      mdspan A = problem.A_view();
      mdspan A_original = problem.A_original_view();
      mdspan x = problem.x_view();
      mdspan A_plus_two_x_xT = problem.A_plus_two_x_xT_view();
      const char what[] = " triangle of A = A + 2.0 x x^T (upper)";

      symmetric_matrix_rank_1_update(2.0, x, A, A, upper_triangle);
      for (int row = 0; row < A.extent(0); ++row) {
        // Lower triangle is unchanged.
        for (int col = 0; col < row; ++col) {
          EXPECT_EQ(A(row, col), A_original(row, col))
            << "A(" << row << "," << col << ") is wrong for "
            << "lower" << what;
        }
        // Upper triangle is changed.
        for (int col = row; col < A.extent(1); ++col) {
          EXPECT_EQ(A(row, col), A_plus_two_x_xT(row, col))
            << "A(" << row << "," << col << ") is wrong for "
            << "upper" << what;
        }
      }
    }
#endif // LINALG_FIX_RANK_UPDATES
  }

  TEST(BLAS3_syr, lower_triangle)
  {
    {
      syr_test_problem problem;
      mdspan A = problem.A_view();
      mdspan A_original = problem.A_original_view();
      mdspan x = problem.x_view();
      mdspan A_plus_x_xT = problem.A_plus_x_xT_view();
      const char what[] = " triangle of A = A + 1.0 x x^T (lower)";

#if defined(LINALG_FIX_RANK_UPDATES)
      symmetric_matrix_rank_1_update(1.0, x, A, A, lower_triangle);
#else
      symmetric_matrix_rank_1_update(1.0, x, A, lower_triangle);
#endif // LINALG_FIX_RANK_UPDATES
      for (int row = 0; row < A.extent(0); ++row) {
        // Lower triangle is changed.
        for (int col = 0; col <= row; ++col) {
          EXPECT_EQ(A(row, col), A_plus_x_xT(row, col))
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
      syr_test_problem problem;
      mdspan A = problem.A_view();
      mdspan A_original = problem.A_original_view();
      mdspan x = problem.x_view();
      mdspan A_plus_x_xT = problem.A_plus_x_xT_view();
      const char what[] = " triangle of A = A + x x^T (lower)";

#if defined(LINALG_FIX_RANK_UPDATES)
      symmetric_matrix_rank_1_update(x, A, A, lower_triangle);
#else
      symmetric_matrix_rank_1_update(x, A, lower_triangle);
#endif // LINALG_FIX_RANK_UPDATES
      for (int row = 0; row < A.extent(0); ++row) {
        // Lower triangle is changed.
        for (int col = 0; col <= row; ++col) {
          EXPECT_EQ(A(row, col), A_plus_x_xT(row, col))
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
      syr_test_problem problem;
      mdspan A = problem.A_view();
      mdspan A_original = problem.A_original_view();
      mdspan x = problem.x_view();
      mdspan x_xT = problem.x_xT_view();
      const char what[] = " triangle of A = x x^T (lower)";

      symmetric_matrix_rank_1_update(x, A, lower_triangle);
      for (int row = 0; row < A.extent(0); ++row) {
        // Lower triangle is changed.
        for (int col = 0; col <= row; ++col) {
          EXPECT_EQ(A(row, col), x_xT(row, col))
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
      syr_test_problem problem;
      mdspan A = problem.A_view();
      mdspan A_original = problem.A_original_view();
      mdspan x = problem.x_view();
      mdspan x_xT = problem.x_xT_view();
      const char what[] = " triangle of A = 1.0 x x^T (lower)";

      symmetric_matrix_rank_1_update(1.0, x, A, lower_triangle);
      for (int row = 0; row < A.extent(0); ++row) {
        // Lower triangle is changed.
        for (int col = 0; col <= row; ++col) {
          EXPECT_EQ(A(row, col), x_xT(row, col))
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
      syr_test_problem problem;
      mdspan A = problem.A_view();
      mdspan A_original = problem.A_original_view();
      mdspan x = problem.x_view();
      mdspan two_x_xT = problem.two_x_xT_view();
      const char what[] = " triangle of A = 2.0 x x^T (lower)";

      symmetric_matrix_rank_1_update(2.0, x, A, lower_triangle);
      for (int row = 0; row < A.extent(0); ++row) {
        // Lower triangle is changed.
        for (int col = 0; col <= row; ++col) {
          EXPECT_EQ(A(row, col), two_x_xT(row, col))
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
      syr_test_problem problem;
      mdspan A = problem.A_view();
      mdspan A_original = problem.A_original_view();
      mdspan x = problem.x_view();
      mdspan A_plus_two_x_xT = problem.A_plus_two_x_xT_view();
      const char what[] = " triangle of A = A + 2.0 x x^T (lower)";

      symmetric_matrix_rank_1_update(2.0, x, A, A, lower_triangle);
      for (int row = 0; row < A.extent(0); ++row) {
        // Lower triangle is changed.
        for (int col = 0; col <= row; ++col) {
          EXPECT_EQ(A(row, col), A_plus_two_x_xT(row, col))
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
