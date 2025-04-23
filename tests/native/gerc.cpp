#include "./gtest_fixtures.hpp"

namespace {
  using LinearAlgebra::lower_triangle;
  using LinearAlgebra::matrix_rank_1_update_c;
  using LinearAlgebra::scaled;
  using LinearAlgebra::upper_triangle;

  // A = [1.0   2.0  3.0]
  //     [4.0  11.0  8.0]
  //     [5.0   3.0  7.0 + 1.0i]
  //
  // x = [  2.0 + 1.0i]
  //     [  5.0 - 3.0i]
  //     [ 11.0 + 5.0i]
  //
  // y = [  3.0 + 1.0i]
  //     [  7.0 - 3.0i]
  //     [ 13.0 + 5.0i]
  class gerc_test_problem {
  public:
    using V = std::complex<double>;

    std::vector<V> A_original{
      V(1.0, 0.0), V( 2.0, 0.0), V(3.0, 0.0),
      V(4.0, 0.0), V(11.0, 0.0), V(8.0, 0.0),
      V(5.0, 0.0), V( 3.0, 0.0), V(7.0, 1.0)
    };

    std::vector<V> A{
      V(1.0, 0.0), V( 2.0, 0.0), V(3.0, 0.0),
      V(4.0, 0.0), V(11.0, 0.0), V(8.0, 0.0),
      V(5.0, 0.0), V( 3.0, 0.0), V(7.0, 1.0)
    };

    std::vector<V> x{
      V( 2.0,  1.0),
      V( 5.0, -3.0),
      V(11.0,  5.0)
    };

    std::vector<V> y{
      V( 3.0,  1.0),
      V( 7.0, -3.0),
      V(13.0,  5.0)
    };

    std::vector<V> A_plus_x_yT = {
      V( 8.0,   1.0), V(13.0,  13.0), V( 34.0,   3.0),
      V(16.0, -14.0), V(55.0,  -6.0), V( 58.0, -64.0),
      V(43.0,   4.0), V(65.0,  68.0), V(175.0,  11.0)
    };

#if defined(LINALG_FIX_RANK_UPDATES)
    std::vector<V> x_yT = {
      V( 7.0,   1.0), V(11.0,  13.0), V( 31.0,   3.0),
      V(12.0, -14.0), V(44.0,  -6.0), V( 50.0, -64.0),
      V(38.0,   4.0), V(62.0,  68.0), V(168.0,  10.0)
    };

    std::vector<V> two_x_yT = {
      V(14.0,   2.0), V( 22.0,  26.0), V( 62.0,    6.0),
      V(24.0, -28.0), V( 88.0, -12.0), V(100.0, -128.0),
      V(76.0,   8.0), V(124.0, 136.0), V(336.0,   20.0)
    };

    std::vector<V> A_plus_two_x_yT = {
      V(15.0,   2.0), V( 24.0,  26.0), V( 65.0,    6.0),
      V(28.0, -28.0), V( 99.0, -12.0), V(108.0, -128.0),
      V(81.0,   8.0), V(127.0, 136.0), V(343.0,   21.0)
    };
#endif // LINALG_FIX_RANK_UPDATES

    using A_type = mdspan<V, extents<int, 3, 3>>;
    using const_A_type = mdspan<const V, extents<int, 3, 3>>;
    using vec_type = mdspan<const V, extents<int, 3>>;
    using result_type = mdspan<const V, extents<int, 3, 3>>;

    A_type A_view() {
      return A_type{A.data()};
    }

    const_A_type A_original_view() const {
      return const_A_type{A_original.data()};
    }

    vec_type x_view() const {
      return vec_type{x.data()};
    }

    vec_type y_view() const {
      return vec_type{y.data()};
    }

    result_type A_plus_x_yT_view() const {
      return result_type{A_plus_x_yT.data()};
    }

#if defined(LINALG_FIX_RANK_UPDATES)
    result_type x_yT_view() const {
      return result_type{x_yT.data()};
    }

    result_type two_x_yT_view() const {
      return result_type{two_x_yT.data()};
    }

    result_type A_plus_two_x_yT_view() const {
      return result_type{A_plus_two_x_yT.data()};
    }
#endif // LINALG_FIX_RANK_UPDATES
  };

  TEST(BLAS3_ger, test0)
  {
    {
      gerc_test_problem problem;
      mdspan A = problem.A_view();
      mdspan A_original = problem.A_original_view();
      mdspan x = problem.x_view();
      mdspan y = problem.y_view();
      mdspan A_plus_x_yT = problem.A_plus_x_yT_view();
      const char what[] = " is wrong for A = A + (1.0 x) y^T";

#if defined(LINALG_FIX_RANK_UPDATES)
      matrix_rank_1_update_c(scaled(1.0, x), y, A, A);
#else
      matrix_rank_1_update_c(scaled(1.0, x), y, A);
#endif // LINALG_FIX_RANK_UPDATES
      for (int row = 0; row < A.extent(0); ++row) {
        for (int col = row; col < A.extent(1); ++col) {
          EXPECT_EQ(A(row, col), A_plus_x_yT(row, col))
            << "A(" << row << "," << col << ")" << what;
        }
      }
    }

    {
      gerc_test_problem problem;
      mdspan A = problem.A_view();
      mdspan A_original = problem.A_original_view();
      mdspan x = problem.x_view();
      mdspan y = problem.y_view();
      mdspan A_plus_x_yT = problem.A_plus_x_yT_view();
      const char what[] = " is wrong for A = A + x y^T";

#if defined(LINALG_FIX_RANK_UPDATES)
      matrix_rank_1_update_c(x, y, A, A);
#else
      matrix_rank_1_update_c(x, y, A);
#endif // LINALG_FIX_RANK_UPDATES
      for (int row = 0; row < A.extent(0); ++row) {
        for (int col = row; col < A.extent(1); ++col) {
          EXPECT_EQ(A(row, col), A_plus_x_yT(row, col))
            << "A(" << row << "," << col << ")" << what;
        }
      }
    }

#if defined(LINALG_FIX_RANK_UPDATES)
    {
      gerc_test_problem problem;
      mdspan A = problem.A_view();
      mdspan A_original = problem.A_original_view();
      mdspan x = problem.x_view();
      mdspan y = problem.y_view();
      mdspan x_yT = problem.x_yT_view();
      const char what[] = " is wrong for A = (1.0 x) y^T";

      matrix_rank_1_update_c(scaled(1.0, x), y, A);
      for (int row = 0; row < A.extent(0); ++row) {
        for (int col = row; col < A.extent(1); ++col) {
          EXPECT_EQ(A(row, col), x_yT(row, col))
            << "A(" << row << "," << col << ")" << what;
        }
      }
    }

    {
      gerc_test_problem problem;
      mdspan A = problem.A_view();
      mdspan A_original = problem.A_original_view();
      mdspan x = problem.x_view();
      mdspan y = problem.y_view();
      mdspan two_x_yT = problem.two_x_yT_view();
      const char what[] = " is wrong for A = (2.0 x) y^T";

      matrix_rank_1_update_c(scaled(2.0, x), y, A);
      for (int row = 0; row < A.extent(0); ++row) {
        for (int col = row; col < A.extent(1); ++col) {
          EXPECT_EQ(A(row, col), two_x_yT(row, col))
            << "A(" << row << "," << col << ")" << what;
        }
      }
    }

    {
      gerc_test_problem problem;
      mdspan A = problem.A_view();
      mdspan A_original = problem.A_original_view();
      mdspan x = problem.x_view();
      mdspan y = problem.y_view();
      mdspan A_plus_two_x_yT = problem.A_plus_two_x_yT_view();
      const char what[] = " is wrong for A = A + (2.0 x) y^T";

      matrix_rank_1_update_c(scaled(2.0, x), y, A, A);
      for (int row = 0; row < A.extent(0); ++row) {
        for (int col = row; col < A.extent(1); ++col) {
          EXPECT_EQ(A(row, col), A_plus_two_x_yT(row, col))
            << "A(" << row << "," << col << ")" << what;
        }
      }
    }
#endif // LINALG_FIX_RANK_UPDATES
  }
} // end anonymous namespace
