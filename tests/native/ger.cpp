#include "./gtest_fixtures.hpp"

namespace {
  using LinearAlgebra::lower_triangle;
  using LinearAlgebra::matrix_rank_1_update;
  using LinearAlgebra::scaled;
  using LinearAlgebra::upper_triangle;

  // A = [1.0   2.0  3.0]
  //     [4.0  11.0  8.0]
  //     [5.0   3.0  7.0]
  //
  // x = [  2.0]
  //     [  5.0]
  //     [ 11.0]
  //
  // y = [  3.0]
  //     [  7.0]
  //     [ 13.0]
  //
  // x y^T = [ 6.0  14.0  26.0]
  //         [15.0  35.0  65.0]
  //         [33.0  77.0 143.0]
  //
  // 2.0 x y^T = [12.0  28.0  52.0]
  //             [30.0  70.0 130.0]
  //             [66.0 154.0 286.0]
  //
  // A + x y^T = [ 7.0  16.0  29.0]
  //             [19.0  46.0  73.0]
  //             [38.0  80.0 150.0]
  //
  // A + 2.0 x y^T = [13.0  30.0  55.0]
  //                 [34.0  88.0 138.0]
  //                 [71.0 157.0 293.0]
  class ger_test_problem {
  public:
    using V = double;

    std::vector<V> A_original{
      V(1.0), V( 2.0), V(3.0),
      V(4.0), V(11.0), V(8.0),
      V(5.0), V( 3.0), V(7.0)
    };

    std::vector<V> A{
      V(1.0), V( 2.0), V(3.0),
      V(4.0), V(11.0), V(8.0),
      V(5.0), V( 3.0), V(7.0)
    };

    std::vector<V> x{
      V( 2.0),
      V( 5.0),
      V(11.0)
    };

    std::vector<V> y{
      V( 3.0),
      V( 7.0),
      V(13.0)
    };

    std::vector<V> A_plus_x_yT = {
      V( 7.0), V(16.0), V( 29.0),
      V(19.0), V(46.0), V( 73.0),
      V(38.0), V(80.0), V(150.0)
    };

#if defined(LINALG_FIX_RANK_UPDATES)
    std::vector<V> x_yT = {
      V( 6.0), V(14.0), V( 26.0),
      V(15.0), V(35.0), V( 65.0),
      V(33.0), V(77.0), V(143.0)
    };

    std::vector<V> two_x_yT = {
      V(12.0), V( 28.0), V( 52.0),
      V(30.0), V( 70.0), V(130.0),
      V(66.0), V(154.0), V(286.0)
    };

    std::vector<V> A_plus_two_x_yT = {
      V(13.0), V( 30.0), V( 55.0),
      V(34.0), V( 81.0), V(138.0),
      V(71.0), V(157.0), V(293.0)
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
      ger_test_problem problem;
      mdspan A = problem.A_view();
      mdspan A_original = problem.A_original_view();
      mdspan x = problem.x_view();
      mdspan y = problem.y_view();
      mdspan A_plus_x_yT = problem.A_plus_x_yT_view();
      const char what[] = " is wrong for A = A + (1.0 x) y^T";

#if defined(LINALG_FIX_RANK_UPDATES)
      matrix_rank_1_update(scaled(1.0, x), y, A, A);
#else
      matrix_rank_1_update(scaled(1.0, x), y, A);
#endif // LINALG_FIX_RANK_UPDATES
      for (int row = 0; row < A.extent(0); ++row) {
        for (int col = row; col < A.extent(1); ++col) {
          EXPECT_EQ(A(row, col), A_plus_x_yT(row, col))
            << "A(" << row << "," << col << ")" << what;
        }
      }
    }

    {
      ger_test_problem problem;
      mdspan A = problem.A_view();
      mdspan A_original = problem.A_original_view();
      mdspan x = problem.x_view();
      mdspan y = problem.y_view();
      mdspan A_plus_x_yT = problem.A_plus_x_yT_view();
      const char what[] = " is wrong for A = A + x y^T";

#if defined(LINALG_FIX_RANK_UPDATES)
      matrix_rank_1_update(x, y, A, A);
#else
      matrix_rank_1_update(x, y, A);
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
      ger_test_problem problem;
      mdspan A = problem.A_view();
      mdspan A_original = problem.A_original_view();
      mdspan x = problem.x_view();
      mdspan y = problem.y_view();
      mdspan x_yT = problem.x_yT_view();
      const char what[] = " is wrong for A = (1.0 x) y^T";

      matrix_rank_1_update(scaled(1.0, x), y, A);
      for (int row = 0; row < A.extent(0); ++row) {
        for (int col = row; col < A.extent(1); ++col) {
          EXPECT_EQ(A(row, col), x_yT(row, col))
            << "A(" << row << "," << col << ")" << what;
        }
      }
    }

    {
      ger_test_problem problem;
      mdspan A = problem.A_view();
      mdspan A_original = problem.A_original_view();
      mdspan x = problem.x_view();
      mdspan y = problem.y_view();
      mdspan two_x_yT = problem.two_x_yT_view();
      const char what[] = " is wrong for A = (2.0 x) y^T";

      matrix_rank_1_update(scaled(2.0, x), y, A);
      for (int row = 0; row < A.extent(0); ++row) {
        for (int col = row; col < A.extent(1); ++col) {
          EXPECT_EQ(A(row, col), two_x_yT(row, col))
            << "A(" << row << "," << col << ")" << what;
        }
      }
    }

    {
      ger_test_problem problem;
      mdspan A = problem.A_view();
      mdspan A_original = problem.A_original_view();
      mdspan x = problem.x_view();
      mdspan y = problem.y_view();
      mdspan A_plus_two_x_yT = problem.A_plus_two_x_yT_view();
      const char what[] = " is wrong for A = A + (2.0 x) y^T";

      matrix_rank_1_update(scaled(2.0, x), y, A, A);
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
